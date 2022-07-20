import torch
import torch.nn.functional as F
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients
from utils.agents import TD3Agent, DDPGAgent
import itertools
import numpy as np
import random

MSELoss = torch.nn.MSELoss()

class MATD3(object):
    def __init__(
        self, 
        agent_init_params, 
        alg_types, 
        adv_init_params=None,
        gamma=0.95, 
        tau=0.01, 
        lr=0.01, 
        hidden_dim=64, 
        discrete_action=False, 
        gaussian_noise_std=None, 
        agent_max_actions=None, 
        cql=False, cql_alpha=None, lse_temp=1.0, num_sampled_actions=None, cql_sample_noise_level=0.2,
        omar=None, omar_coe=None,
        omar_mu=None, omar_sigma=None, omar_num_samples=None, omar_num_elites=None, omar_iters=None, batch_size=None, 
        env_id=None,
    ):
        self.env_id = env_id
        self.is_mamujoco = True if self.env_id == 'HalfCheetah-v2' else False

        assert (ma == agent_max_actions[0] for ma in agent_max_actions)
        self.max_action = agent_max_actions[0]
        self.min_action = -self.max_action

        self.nagents = len(alg_types)
        self.alg_types = alg_types

        self.agents = [TD3Agent(
            lr=lr, 
            discrete_action=discrete_action, 
            hidden_dim=hidden_dim, 
            gaussian_noise_std=gaussian_noise_std, 
            **params
        ) for params in agent_init_params]

        if self.env_id in ['simple_tag', 'simple_world']:
            self.num_predators = len(agent_init_params)
            self.num_preys = len(adv_init_params)

            self.preys = [DDPGAgent(lr=lr, discrete_action=discrete_action, hidden_dim=hidden_dim, **params) for params in adv_init_params]

        self.niter = 0

        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action

        self.pol_dev, self.trgt_pol_dev, self.critic_dev, self.trgt_critic_dev = 'cpu', 'cpu', 'cpu', 'cpu' 

        self.omar = omar
        if self.omar:
            self.omar_coe = omar_coe

            self.omar_iters = omar_iters
            self.omar_num_samples = omar_num_samples
            self.init_omar_mu, self.init_omar_sigma = omar_mu, omar_sigma
            self.omar_mu = torch.cuda.FloatTensor(batch_size, self.agent_init_params[0]['num_out_pol']).zero_() + self.init_omar_mu
            self.omar_sigma = torch.cuda.FloatTensor(batch_size, self.agent_init_params[0]['num_out_pol']).zero_() + self.init_omar_sigma
            self.omar_num_elites = omar_num_elites

        self.cql = cql
        if self.cql:
            self.cql_alpha = cql_alpha
            self.cql_sample_noise_level = cql_sample_noise_level
            self.lse_temp = lse_temp
            self.num_sampled_actions = num_sampled_actions

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        res = []
        for i, obs in zip(range(self.nagents), observations):
            if self.env_id in ['simple_world', 'simple_tag']:
                if i < self.num_predators:
                    predator_action = self.agents[i].step(obs, explore=explore)
                    res.append(predator_action)
                else:
                    prey_action = self.preys[i - self.num_predators].step(obs, explore=False)
                    res.append(prey_action)
            else:
                action = self.agents[i].step(obs, explore=explore)
                res.append(action)
        return res

    def calc_gaussian_pdf(self, samples, mu=0):
        pdfs = 1 / (self.cql_sample_noise_level * np.sqrt(2 * np.pi)) * torch.exp( - (samples - mu)**2 / (2 * self.cql_sample_noise_level**2) )
        pdf = torch.prod(pdfs, dim=-1)
        return pdf

    def get_policy_actions(self, state, network):
        action = network(state)

        formatted_action = action.unsqueeze(1).repeat(1, self.num_sampled_actions, 1).view(action.shape[0] * self.num_sampled_actions, action.shape[1])

        random_noises = torch.FloatTensor(formatted_action.shape[0], formatted_action.shape[1])

        random_noises = random_noises.normal_() * self.cql_sample_noise_level
        random_noises_log_pi = self.calc_gaussian_pdf(random_noises).view(action.shape[0], self.num_sampled_actions, 1).cuda()
        random_noises = random_noises.cuda()

        noisy_action = (formatted_action + random_noises).clamp(-self.max_action, self.max_action)

        return noisy_action, random_noises_log_pi

    def compute_softmax_acs(self, q_vals, acs):
        max_q_vals = torch.max(q_vals, 1, keepdim=True)[0]
        norm_q_vals = q_vals - max_q_vals
        e_beta_normQ = torch.exp(norm_q_vals)
        a_mult_e = acs * e_beta_normQ
        numerators = a_mult_e
        denominators = e_beta_normQ

        sum_numerators = torch.sum(numerators, 1)
        sum_denominators = torch.sum(denominators, 1)

        softmax_acs = sum_numerators / sum_denominators

        return softmax_acs

    def update(self, sample, agent_i, parallel=False):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next observations, and episode end masks) 
                    sampled randomly from the replay buffer. Each is a list with entries corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch): If passed in, important quantities will be logged
        """
        if self.is_mamujoco:
            states, obs, acs, rews, next_states, next_obs, dones = sample
        else:
            obs, acs, rews, next_obs, dones = sample

        curr_agent = self.agents[agent_i]

        curr_agent.critic_optimizer.zero_grad()
        
        trgt_acs = curr_agent.target_policy(next_obs[agent_i])
        trgt_vf_in = torch.cat((next_obs[agent_i], trgt_acs), dim=1)
        
        next_q_value1, next_q_value2 = curr_agent.target_critic(trgt_vf_in) 
        next_q_value = torch.min(next_q_value1, next_q_value2)

        target_value = rews[agent_i].view(-1, 1) + self.gamma * next_q_value * (1 - dones[agent_i].view(-1, 1)) 

        vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
        
        actual_value1, actual_value2 = curr_agent.critic(vf_in) 

        vf_loss = MSELoss(actual_value1, target_value.detach()) + MSELoss(actual_value2, target_value.detach())
        
        if self.cql:
            if self.is_mamujoco:
                formatted_obs = obs[agent_i].unsqueeze(1).repeat(1, self.num_sampled_actions, 1).view(-1, obs[agent_i].shape[1])

                random_action = (torch.FloatTensor(acs[agent_i].shape[0] * self.num_sampled_actions, acs[agent_i].shape[1]).uniform_(-1, 1)).cuda()
                random_action_log_pi = np.log(0.5 ** random_action.shape[-1])
                curr_action, curr_action_log_pi = self.get_policy_actions(obs[agent_i], curr_agent.policy)
                new_curr_action, new_curr_action_log_pi = self.get_policy_actions(next_obs[agent_i], curr_agent.policy)

                random_vf_in = torch.cat((formatted_obs, random_action), dim=1)
                curr_vf_in = torch.cat((formatted_obs, curr_action), dim=1)
                new_curr_vf_in = torch.cat((formatted_obs, new_curr_action), dim=1)

                random_Q1, random_Q2 = curr_agent.critic(random_vf_in)
                curr_Q1, curr_Q2 = curr_agent.critic(curr_vf_in)
                new_curr_Q1, new_curr_Q2 = curr_agent.critic(new_curr_vf_in)

                random_Q1, random_Q2 = random_Q1.view(obs[agent_i].shape[0], self.num_sampled_actions, 1), random_Q2.view(obs[agent_i].shape[0], self.num_sampled_actions, 1)
                curr_Q1, curr_Q2 = curr_Q1.view(obs[agent_i].shape[0], self.num_sampled_actions, 1), curr_Q2.view(obs[agent_i].shape[0], self.num_sampled_actions, 1)
                new_curr_Q1, new_curr_Q2 = new_curr_Q1.view(obs[agent_i].shape[0], self.num_sampled_actions, 1), new_curr_Q2.view(obs[agent_i].shape[0], self.num_sampled_actions, 1)

                cat_q1 = torch.cat([random_Q1 - random_action_log_pi, new_curr_Q1 - new_curr_action_log_pi, curr_Q1 - curr_action_log_pi], 1)
                cat_q2 = torch.cat([random_Q2 - random_action_log_pi, new_curr_Q2 - new_curr_action_log_pi, curr_Q2 - curr_action_log_pi], 1)
                
                policy_qvals1 = torch.logsumexp(cat_q1 / self.lse_temp, dim=1) * self.lse_temp
                policy_qvals2 = torch.logsumexp(cat_q2 / self.lse_temp, dim=1) * self.lse_temp
            else:
                formatted_obs = obs[agent_i].unsqueeze(1).repeat(1, self.num_sampled_actions, 1).view(-1, obs[agent_i].shape[1])

                random_acs = (torch.FloatTensor(acs[agent_i].shape[0] * self.num_sampled_actions, acs[agent_i].shape[1]).uniform_(-1, 1)).cuda()
                random_acs_log_pi = np.log(0.5 ** random_acs.shape[-1])

                random_vf_in = torch.cat((formatted_obs, random_acs), dim=1)

                random_qvals1, random_qvals2 = curr_agent.critic(random_vf_in)

                random_qvals1 = random_qvals1.view(obs[agent_i].shape[0], self.num_sampled_actions)
                random_qvals2 = random_qvals2.view(obs[agent_i].shape[0], self.num_sampled_actions)

                policy_qvals1 = torch.logsumexp((random_qvals1 - random_acs_log_pi) / self.lse_temp, dim=1, keepdim=True) * self.lse_temp 
                policy_qvals2 = torch.logsumexp((random_qvals2 - random_acs_log_pi) / self.lse_temp, dim=1, keepdim=True) * self.lse_temp

            dataset_q_vals1 = actual_value1
            dataset_q_vals2 = actual_value2

            cql_term1 = (policy_qvals1 - dataset_q_vals1).mean()
            cql_term2 = (policy_qvals2 - dataset_q_vals2).mean()
            
            cql_term = cql_term1 + cql_term2
            vf_loss += self.cql_alpha * cql_term

        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()


        curr_agent.policy_optimizer.zero_grad()

        curr_pol_out = curr_agent.policy(obs[agent_i])
        curr_pol_vf_in = curr_pol_out
        
        vf_in = torch.cat((obs[agent_i], curr_pol_vf_in), dim=1)

        if self.omar:
            pred_qvals = curr_agent.critic.Q1(vf_in)

            if self.is_mamujoco:
                self.omar_mu = torch.cuda.FloatTensor(acs[agent_i].shape[0], acs[agent_i].shape[1]).zero_() + self.init_omar_mu
                self.omar_sigma = torch.cuda.FloatTensor(acs[agent_i].shape[0], acs[agent_i].shape[1]).zero_() + self.init_omar_sigma

                formatted_obs = obs[agent_i].unsqueeze(1).repeat(1, self.omar_num_samples, 1).view(-1, obs[agent_i].shape[1])

                last_top_k_qvals, last_elite_acs = None, None
                for iter_idx in range(self.omar_iters):
                    dist = torch.distributions.Normal(self.omar_mu, self.omar_sigma)
                    
                    cem_sampled_acs = dist.sample((self.omar_num_samples,)).detach().permute(1, 0, 2).clamp(-self.max_action, self.max_action)

                    formatted_cem_sampled_acs = cem_sampled_acs.view(-1, cem_sampled_acs.shape[-1])

                    vf_in = torch.cat((formatted_obs, formatted_cem_sampled_acs), dim=1)
                    all_pred_qvals = curr_agent.critic.Q1(vf_in).view(acs[agent_i].shape[0], -1, 1)

                    if iter_idx > 0:
                        all_pred_qvals = torch.cat((all_pred_qvals, last_top_k_qvals), dim=1)
                        cem_sampled_acs = torch.cat((cem_sampled_acs, last_elite_acs), dim=1)

                    top_k_qvals, top_k_inds = torch.topk(all_pred_qvals, self.omar_num_elites, dim=1)
                    elite_ac_inds = top_k_inds.repeat(1, 1, acs[agent_i].shape[1])
                    elite_acs = torch.gather(cem_sampled_acs, 1, elite_ac_inds)

                    last_top_k_qvals, last_elite_acs = top_k_qvals, elite_acs

                    updated_mu = torch.mean(elite_acs, dim=1)
                    updated_sigma = torch.std(elite_acs, dim=1)

                    self.omar_mu = updated_mu
                    self.omar_sigma = updated_sigma

                top_qvals, top_inds = torch.topk(all_pred_qvals, 1, dim=1)
                top_ac_inds = top_inds.repeat(1, 1, acs[agent_i].shape[1])
                top_acs = torch.gather(cem_sampled_acs, 1, top_ac_inds)

                cem_qvals = top_qvals
                pol_qvals = pred_qvals.unsqueeze(1)
                cem_acs = top_acs
                pol_acs = curr_pol_out.unsqueeze(1)

                candidate_qvals = torch.cat([pol_qvals, cem_qvals], 1)
                candidate_acs = torch.cat([pol_acs, cem_acs], 1)

                max_qvals, max_inds = torch.max(candidate_qvals, 1, keepdim=True)
                max_ac_inds = max_inds.repeat(1, 1, acs[agent_i].shape[1])

                max_acs = torch.gather(candidate_acs, 1, max_ac_inds).squeeze(1)
            else:
                self.omar_mu = torch.cuda.FloatTensor(acs[agent_i].shape[0], acs[agent_i].shape[1]).zero_() + self.init_omar_mu
                self.omar_sigma = torch.cuda.FloatTensor(acs[agent_i].shape[0], acs[agent_i].shape[1]).zero_() + self.init_omar_sigma

                formatted_obs = obs[agent_i].unsqueeze(1).repeat(1, self.omar_num_samples, 1).view(-1, obs[agent_i].shape[1])

                for iter_idx in range(self.omar_iters):
                    dist = torch.distributions.Normal(self.omar_mu, self.omar_sigma)

                    cem_sampled_acs = dist.sample((self.omar_num_samples,)).detach().permute(1, 0, 2).clamp(-self.max_action, self.max_action)

                    formatted_cem_sampled_acs = cem_sampled_acs.view(-1, cem_sampled_acs.shape[-1])

                    vf_in = torch.cat((formatted_obs, formatted_cem_sampled_acs), dim=1)
                    all_pred_qvals = curr_agent.critic.Q1(vf_in)
                    all_pred_qvals = all_pred_qvals.view(acs[agent_i].shape[0], -1, 1)

                    updated_mu = self.compute_softmax_acs(all_pred_qvals, cem_sampled_acs)
                    self.omar_mu = updated_mu

                    updated_sigma = torch.sqrt(torch.mean((cem_sampled_acs - updated_mu.unsqueeze(1)) ** 2, 1))
                    self.omar_sigma = updated_sigma

                top_qvals, top_inds = torch.topk(all_pred_qvals, 1, dim=1)
                top_ac_inds = top_inds.repeat(1, 1, acs[agent_i].shape[1])
                top_acs = torch.gather(cem_sampled_acs, 1, top_ac_inds)

                cem_qvals = top_qvals
                pol_qvals = pred_qvals.unsqueeze(1)
                cem_acs = top_acs
                pol_acs = curr_pol_out.unsqueeze(1)

                candidate_qvals = torch.cat([pol_qvals, cem_qvals], 1)
                candidate_acs = torch.cat([pol_acs, cem_acs], 1)

                max_qvals, max_inds = torch.max(candidate_qvals, 1, keepdim=True)
                max_ac_inds = max_inds.repeat(1, 1, acs[agent_i].shape[1])

                max_acs = torch.gather(candidate_acs, 1, max_ac_inds).squeeze(1)
                        
            mimic_acs = max_acs.detach()
            
            mimic_term = F.mse_loss(curr_pol_out, mimic_acs)

            pol_loss = self.omar_coe * mimic_term - (1 - self.omar_coe) * pred_qvals.mean()
        else:
            pol_loss = -curr_agent.critic.Q1(vf_in).mean()
        
        pol_loss += (curr_pol_out ** 2).mean() * 1e-3
        
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been performed for each agent)
        """
        if self.env_id in ['simple_tag', 'simple_world']:
            end_idx = self.num_predators
        else:
            end_idx = len(self.agents)

        for i, a in enumerate(self.agents[:end_idx]):
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau) 
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train() 
            a.target_critic.train()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)

            if self.env_id in ['simple_tag', 'simple_world']:
                for p in self.preys:
                    p.policy = fn(p.policy)

            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device: 
            for a in self.agents: 
                a.target_policy = fn(a.target_policy)

            if self.env_id in ['simple_tag', 'simple_world']:
                for p in self.preys:
                    p.target_policy = fn(p.target_policy)

            self.trgt_pol_dev = device 
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            if self.env_id in ['simple_tag', 'simple_world']:
                for p in self.preys:
                    p.policy = fn(p.policy)
            self.pol_dev = device

    @classmethod
    def init_from_env(cls, env, env_id, data_type, env_info=None, agent_alg="td3", adversary_alg="ddpg", gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, cql=False, batch_size=None, lse_temp=None, num_sampled_actions=None, gaussian_noise_std=None, omar=None, omar_mu=None, omar_sigma=None, omar_num_samples=None, omar_num_elites=None, omar_iters=None):
        """
        Instantiate instance of this class from multi-agent environment
        """
        if env_id in ['simple_tag', 'simple_world']:
            alg_types = [agent_alg if atype == 'adversary' else adversary_alg for atype in env.agent_types]
        elif env_id in ['simple_spread']:
            alg_types = [agent_alg for atype in env.agent_types]
        elif env_id in ['HalfCheetah-v2']:
            alg_types = [agent_alg for _ in range(env_info['n_agents'])]

        agent_init_params = []
        all_n_actions = []
        agent_max_actions = []
        adv_init_params = []

        if env_id == 'HalfCheetah-v2':
            for agent_idx, algtype in zip(range(len(alg_types)), alg_types):
                acsp = env_info['action_spaces'][agent_idx]

                num_in_pol = env_info['obs_shape']
                num_out_pol = acsp.shape[0]
                num_in_critic = num_in_pol + num_out_pol

                agent_init_params.append({'num_in_pol': num_in_pol, 'num_out_pol': num_out_pol, 'num_in_critic': num_in_critic})
                
                agent_max_actions.append(acsp.high[0])
                all_n_actions.append(acsp.shape[0])
        else:
            for acsp, obsp, algtype, agent_type in zip(env.action_space, env.observation_space, alg_types, env.agent_types):
                num_in_pol = obsp.shape[0]
                num_out_pol = acsp.shape[0]
                num_in_critic = num_in_pol + num_out_pol

                if env_id in ['simple_spread']:
                    agent_init_params.append({'num_in_pol': num_in_pol, 'num_out_pol': num_out_pol, 'num_in_critic': num_in_critic})
                    agent_max_actions.append(acsp.high[0])
                else:
                    if agent_type == 'adversary':
                        agent_init_params.append({'num_in_pol': num_in_pol, 'num_out_pol': num_out_pol, 'num_in_critic': num_in_critic})
                        agent_max_actions.append(acsp.high[0])
                    elif agent_type == 'agent':
                        adv_init_params.append({'num_in_pol': num_in_pol, 'num_out_pol': num_out_pol, 'num_in_critic': num_in_critic})

                all_n_actions.append(acsp.shape[0])

            for i in range(1, len(all_n_actions)):
                assert (all_n_actions[i] == all_n_actions[0])

        env_config_map = {
            'simple_spread': {
                'random': {'omar_coe': 1.0, 'cql_alpha': 0.5},
                'medium-replay': {'omar_coe': 1.0, 'cql_alpha': 1.0},
                'medium': {'omar_coe': 1.0, 'cql_alpha': 5.0},
                'expert': {'omar_coe': 1.0, 'cql_alpha': 5.0},
            },
            'simple_tag': {
                'random': {'omar_coe': 0.9, 'cql_alpha': 0.5},
                'medium-replay': {'omar_coe': 0.9, 'cql_alpha': 0.5},
                'medium': {'omar_coe': 0.7, 'cql_alpha': 5.0},
                'expert': {'omar_coe': 0.9, 'cql_alpha': 5.0},
            },
            'simple_world': {
                'random': {'omar_coe': 1.0, 'cql_alpha': 0.5},
                'medium-replay': {'omar_coe': 0.7, 'cql_alpha': 1.0},
                'medium': {'omar_coe': 0.1, 'cql_alpha': 0.5},
                'expert': {'omar_coe': 0.9, 'cql_alpha': 5.0},
            },
            'HalfCheetah-v2': {
                'random': {'omar_coe': 1.0, 'cql_alpha': 1.0},
                'medium-replay': {'omar_coe': 0.9, 'cql_alpha': 5.0},
                'medium': {'omar_coe': 0.7, 'cql_alpha': 1.0},
                'expert': {'omar_coe': 0.5, 'cql_alpha': 5.0},
            }
        }
        omar_coe = env_config_map[env_id][data_type]['omar_coe']
        cql_alpha = env_config_map[env_id][data_type]['cql_alpha']
        cql = True if omar else cql

        init_dict = {
            'env_id': env_id,
            'gamma': gamma, 
            'tau': tau, 
            'lr': lr,
            'hidden_dim': hidden_dim,
            'alg_types': alg_types,
            'agent_init_params': agent_init_params,
            'adv_init_params': adv_init_params, 
            'discrete_action': False,
            'cql': cql, 'cql_alpha': cql_alpha, 'lse_temp': lse_temp, 'num_sampled_actions': num_sampled_actions,
            'batch_size': batch_size,
            'gaussian_noise_std': gaussian_noise_std,
            'agent_max_actions': agent_max_actions,
            'omar': omar, 'omar_coe': omar_coe,
            'omar_iters': omar_iters, 'omar_mu': omar_mu, 'omar_sigma': omar_sigma, 'omar_num_samples': omar_num_samples, 'omar_num_elites': omar_num_elites,
        }
        
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        
        return instance

    def load_pretrained_preys(self, filename):
        if not torch.cuda.is_available():
            save_dict = torch.load(filename, map_location=torch.device('cpu'))
        else:
            save_dict = torch.load(filename)

        if self.env_id in ['simple_tag', 'simple_world']:
            prey_params = save_dict['agent_params'][self.num_predators:]

        for i, params in zip(range(self.num_preys), prey_params):
            self.preys[i].load_params_without_optims(params)

        for p in self.preys:
            p.policy.eval()
            p.target_policy.eval()

