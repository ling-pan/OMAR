import argparse
import torch
import time
import os, sys, tempfile
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from algorithms.maddpg import MATD3
import datetime
import random
import copy
import shutil

from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
import h5py

try:
    from multiagent_mujoco.src.multiagent_mujoco.mujoco_multi import MujocoMulti
except:
    print ('MujocoMulti not installed')

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def eval_policy(agent, env_name, seed, eval_episodes, discrete_action, env_args=None):
    if env_name in ['HalfCheetah-v2']:
        env = MujocoMulti(env_args=env_args)
        env.seed(seed + 100)

        all_episodes_rewards = []
        for ep_i in range(eval_episodes):
            agent.prep_rollouts(device='cpu')

            env.reset()
            done = False
            episode_reward = 0.
            while not done:
                obs = env.get_obs()

                torch_obs = [Variable(torch.Tensor(obs[i]).unsqueeze(0), requires_grad=False) for i in range(agent.nagents)] 
                torch_agent_actions = agent.step(torch_obs, explore=False)
                agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                actions = [ac.squeeze(0) for ac in agent_actions]

                reward, done, info = env.step(actions)

                episode_reward += reward

            all_episodes_rewards.append(episode_reward)
        
        mean_episode_reward = np.mean(np.array(all_episodes_rewards))
        return mean_episode_reward
    else:
        avg_predator_return = 0.
    
        env = make_parallel_env(env_name, 1, seed + 100, discrete_action)

        for ep_i in range(0, eval_episodes):
            obs = env.reset()
            agent.prep_rollouts(device='cpu')

            for et_i in range(config.episode_length):
                torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False) for i in range(agent.nagents)]
                torch_agent_actions = agent.step(torch_obs, explore=False)
                agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
                
                next_obs, rewards, dones, infos = env.step(actions)
                
                if env_name in ['simple_tag', 'simple_world']:
                    avg_predator_return += rewards[0][0]
                else:
                    avg_agent_reward = np.mean(rewards[0])
                    avg_predator_return += avg_agent_reward

                obs = next_obs

        avg_predator_return /= eval_episodes
        return avg_predator_return

def offline_train(config):
    outdir = prepare_output_dir(config.dir + '/' + config.env_id, argv=sys.argv)
    print('\033[1;32mOutput files are saved in {} \033[1;0m'.format(outdir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.set_num_threads(config.n_training_threads)

    if config.env_id in ['simple_spread', 'simple_tag', 'simple_world']:
        env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed, config.discrete_action)
        env_args, env_info = None, None
    else:
        env_args = {"scenario": config.env_id, "episode_limit": 1000, "agent_conf": '2x3', "agent_obsk": 0,}
        env = MujocoMulti(env_args=env_args)
        env.seed(config.seed)

        env_info = env.get_env_info()

        config.batch_size = 256
        config.hidden_dim = 256
        config.lr = 0.0003
        config.tau = 0.005
        config.gamma = 0.99

        config.omar_iters = 2
        config.omar_num_samples = 20
        config.omar_num_elites = 5 

    ma_agent = MATD3.init_from_env(
        env, config.env_id, config.data_type,
        tau=config.tau, lr=config.lr, hidden_dim=config.hidden_dim,
        cql=config.cql, lse_temp=config.lse_temp, batch_size=config.batch_size, num_sampled_actions=config.num_sampled_actions,
        omar=config.omar, omar_iters=config.omar_iters, omar_mu=config.omar_mu, omar_sigma=config.omar_sigma, omar_num_samples=config.omar_num_samples, omar_num_elites=config.omar_num_elites, 
        env_info=env_info, 
    )

    if config.env_id in ['simple_tag', 'simple_world']:
        pretrained_model_dir = './datasets/{}/pretrained_adv_model.pt'.format(config.env_id)
        ma_agent.load_pretrained_preys(pretrained_model_dir)

    if config.env_id in ['simple_spread', 'simple_tag', 'simple_world']:
        replay_buffer = ReplayBuffer(
            config.buffer_length, ma_agent.nagents,
            [obsp.shape[0] for obsp in env.observation_space],
            [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_space],
        )
    else:
        replay_buffer = ReplayBuffer(
            config.buffer_length, ma_agent.nagents,
            [env_info['obs_shape'] for _ in env.observation_space],
            [acsp.shape[0] for acsp in env.action_space],
            is_mamujoco=True,
            state_dims=[env_info['state_shape'] for _ in env.observation_space],
        )
    replay_buffer.load_batch_data(config.dataset_dir)

    for t in range(config.num_steps + 1):
        if t % config.eval_interval == 0 or t == config.num_steps:
            eval_return = eval_policy(ma_agent, config.env_id, config.seed, config.eval_episodes, config.discrete_action, env_args=env_args)

        if (t % config.steps_per_update) < config.n_rollout_threads:
            ma_agent.prep_training(device='gpu') if config.use_gpu else ma_agent.prep_training(device='cpu')

            for u_i in range(config.n_rollout_threads):
                nagents = ma_agent.nagents if config.env_id in ['simple_spread', 'HalfCheetah-v2'] else ma_agent.num_predators

                for a_i in range(nagents):
                    sample = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)

                    ma_agent.update(sample, a_i)

                ma_agent.update_all_targets()

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Name of directory to store model/training contents", type=str, default='results')

    parser.add_argument("--env_id", help="Name of environment", type=str, default='simple_spread')
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=1, type=int)
    parser.add_argument("--discrete_action", action='store_true', default=False)
    parser.add_argument("--use_gpu", default=1, type=int)

    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size for model training")
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument('--num_updates', default=1, type=int)
    parser.add_argument("--gamma", default=0.95, type=float)

    parser.add_argument('--gaussian_noise_std', default=0.1, type=float)

    parser.add_argument("--data_type", default='medium', type=str)
    parser.add_argument('--dataset_dir', default='./datasets', type=str)

    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--eval_interval', default=1000, type=int)
    parser.add_argument('--num_steps', default=int(1e5), type=int)

    parser.add_argument('--cql', default=0, type=int)
    parser.add_argument('--cql_alpha', default=1.0, type=float)
    parser.add_argument("--lse_temp", default=1.0, type=float)
    parser.add_argument('--num_sampled_actions', default=10, type=int) 
    parser.add_argument('--cql_sample_noise_level', default=0.2, type=float)

    parser.add_argument('--omar', default=0, type=int) 
    parser.add_argument('--omar_coe', default=1.0, type=float) 
    parser.add_argument('--omar_iters', default=3, type=int)
    parser.add_argument('--omar_mu', default=0., type=float)
    parser.add_argument('--omar_sigma', default=2.0, type=float)
    parser.add_argument('--omar_num_samples', default=10, type=int)
    parser.add_argument('--omar_num_elites', default=10, type=int)
    config = parser.parse_args()

    if config.env_id in ['simple_spread', 'simple_tag']:
        config.num_steps = 200000
        if config.env_id == 'simple_spread' and config.data_type == 'random':
            config.num_steps = 600000
        elif config.env_id == 'simple_tag' and config.data_type == 'medium-replay':
            config.num_steps = 100000
    elif config.env_id == 'HalfCheetah-v2':
        config.num_steps = int(1e6)
        config.steps_per_update = 10
        config.eval_interval = 5000
        
    config.dataset_dir = config.dataset_dir + '/' + config.env_id + '/' + config.data_type + '/' + 'seed_{}_data'.format(config.seed)
        
    offline_train(config)