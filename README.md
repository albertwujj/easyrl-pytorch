# easyrl
Reinforcement learning algorithms in Pytorch. Thouroughly commented, clear implementations.
## Proximal Policy Optimization
An RL algorithm where the maximization objective given a state-action pair is the advantage times ratio of the probability over the old probability, clipped ([paper](https://arxiv.org/pdf/1707.06347.pdf)).  
Works in parallel with multiple envs.
