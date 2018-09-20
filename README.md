# easyrl
Reinforcement learning algorithms in Pytorch. Thouroughly commented, clear implementations.
## Proximal Policy Optimization
RL algorithm where the maximization objective given a state-action pair is the advantage times ratio of the action probability over the old action probability, clipped ([paper](https://arxiv.org/pdf/1707.06347.pdf)).  

Works with any environment with discrete actions. Works with multiple envs in parallel. Tested on OpenAI Retro's Sonic environment.
