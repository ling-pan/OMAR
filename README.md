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

## Datasets
Datasets for different tasks are available at the following links. Please download the datasets and decompress them to the datasets folder.
- [HalfCheetah](https://drive.google.com/file/d/1zELoWUZoy3wPpwYni9t_TbzOjF4Px2f0/view?usp=sharing)
- [Cooperative Navigation](https://pan.baidu.com/s/1QtyCSFAiSH9yn3dSiCP6gA): password is teve
- [Predator-Prey](https://pan.baidu.com/s/16W-UyyCtfKDt9oTgeNOhJA): password is m7vw
- [World](https://pan.baidu.com/s/1pjZmeIAlaepPpug3b5olGA): password is 5k3t

Note: The datasets are too large, and the Baidu (Chinese) online disk requires a password for accessing it. Please just enter the password in the input box and click the blue button. The dataset can then be downloaded by cliking the "download" button (the second white button).

## Usage

Please follow the instructions below to replicate the results in the paper. 

```
pythonmain.py --env_id <ENVIRONMENT_NAME> --data_type <DATA_TYPE> --seed <SEED> --omar 1
```

- env_id: simple_spread/tag/world/HalfCheetah-v2
- data_type: random/medium-replay/medium/expert

