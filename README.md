# Plan Better Amid Conservatism: Offline Multi-Agent Reinforcement Learning with Actor Rectification

This repository is the implementation of [Plan Better Amid Conservatism: Offline Multi-Agent Reinforcement Learning with Actor Rectification](https://proceedings.mlr.press/v162/pan22a/pan22a.pdf) in ICML 2022. This codebase is based on the open-source [maddpg-pytorch](https://github.com/shariqiqbal2810/maddpg-pytorch) framework, and please refer to that repo for more documentation.

## Citing

If you used this code in your research or found it helpful, please consider citing our paper:
Bibtex:

```
@inproceedings{pan2021regularized,
  title={Plan Better Amid Conservatism: Offline Multi-Agent Reinforcement Learning with Actor Rectification},
  author={Pan, Ling and Huang, Longbo and Ma, Tengyu and Xu, Huazhe},
  booktitle={International Conference on Machine
Learning},
  year={2022}
}
```

## Requirements

- Multi-agent Particle Environments: in envs/multiagent-particle-envs and install it by `pip install -e .`
- python: 3.6
- torch
- baselines (https://github.com/openai/baselines)
- seaborn
- gym==0.9.4
- Multi-Agent MuJoCo: Please check the [multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco) repo for more details about the environment. Note that this depends on gym with version 0.10.5.

## Usage

Please follow the instructions below to replicate the results in the paper. 

```
pythonmain.py --env_id <ENVIRONMENT_NAME> --data_type <DATA_TYPE> --seed <SEED> --omar 1
```

- env_id: simple_spread/tag/world, HalfCheetah-v2
- data_type: random/medium-replay/medium/expert

