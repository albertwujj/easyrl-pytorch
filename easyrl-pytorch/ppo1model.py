import torch
import torch.nn as nn
import numpy as np
import gym
import random
import math
import time

import gym

""" PURPOSE
# RL implementation is very simple math. Let's not make it harder than it needs to be. There are two functional differences between this PPO
# and OpenAI's very opaque implementation: 

This only runs on one environment at a time. In order to handle multiple, add a dimension to the arrays in the Runner class,
along with using an array to keep track of which environments are done. 

This does not work with recurrent neural networks. You would need to track a history of states.
"""

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# The neural network layers shared by both action and value function
class ConvNet(nn.Module):
    def __init__(self, obs_length, num_actions):

        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(20800, 10)
        self.fcAction = nn.Linear(10, num_actions)
        self.fcValue = nn.Linear(10, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return self.fcValue(out), self.fcAction(out)


def calculateEntropy(logits):
    a0 = logits - torch.max(logits, -1, keepdim=True)[0]
    ea0 = torch.exp(a0)
    z0 = torch.sum(logits, -1, keepdim=True)
    p0 = ea0 / z0
    return torch.sum(p0 * (torch.log(z0) - a0), -1)

class Model():

    def __init__(self, obs_space, ac_space, nsteps, vf_coef, lr):
       self.nn = ConvNet(obs_space, ac_space).to(device)
       self.obs_space, self.ac_space, self.nsteps, self.vf_coef \
           = obs_space, ac_space, nsteps, vf_coef
       self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=lr)
       self.v_loss_weight = vf_coef

    # each tensor can actually represent more than 1 step. First dimension is step #
    def train(self, obs, v_prev, v_target, action_index, a_logit_prev, sum_exp_logits_prev, cliprange):
        # CALCULATE PPO LOSS
        value, a_logits = self.nn(obs)

        v_clipped = torch.clamp(value - v_prev, -cliprange, cliprange) + v_prev
        v_loss = (value - v_target) ** 2
        v_loss_clipped = (v_clipped - v_target) ** 2
        # model will minimize the clipped or the non-clipped loss, whichever is greater
        v_loss_final = .5 * torch.mean(torch.max(v_loss, v_loss_clipped))

        sum_exp_logits = torch.sum(torch.exp(a_logits), -1)
        entropy = torch.mean(calculateEntropy(a_logits))
        # a_logit is the unscaled log of the actual action probability (a_prob), as no softmax has been applied.
        selected_a_logit = a_logits[np.arange(a_logits.shape[0]),[i for i in action_index]]

        adv = v_target - v_prev


        # equivalent to dividing the actual (after softmax) predicted prob by the previous predicted prob
        ratio = torch.exp(selected_a_logit - a_logit_prev) / sum_exp_logits * sum_exp_logits_prev
        print(sum_exp_logits)
        print(sum_exp_logits_prev)
        print(ratio)
        a_loss = - adv * ratio
        a_loss_clipped = - adv * torch.clamp(ratio, 1.0 - cliprange, cliprange)
        a_loss_final = .5 * torch.mean(torch.max(a_loss, a_loss_clipped))

        loss = a_loss_final + v_loss_final * self.v_loss_weight - entropy * 0.01

        # GRADIENT DESCENT
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True) # compute gradients
        self.optimizer.step() # apply gradients

        return loss.item()

    def evaluate(self, obs_tensor):
        value, a_logits = self.nn(obs_tensor)
        value=torch.squeeze(value, dim=0)
        a_logits = torch.squeeze(a_logits, dim=0)
        sum_exp_logits = torch.sum(torch.exp(a_logits), -1)
        a_logit, action_index = a_logits.max(0)
        return action_index.item(), value, a_logit, sum_exp_logits

# This class will follow output from the model to take actions in the environment.
# It will return the tuples of experience (observation, action, reward) it receives
# along with the calculated advantages
class Runner(object):
    def __init__(self, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        # initial observations

        # parameters for calculating the advantage of a state-action pair
        self.gamma = gamma
        self.lam = lam

        self.nsteps = nsteps

    def run(self):
        # the experience to return (to eventually send to the model)
        stored_obs, stored_rewards, stored_actions, stored_vpreds, stored_neglogprobs, stored_se_logits = [], [], [], [], [], []

        step_time = 0
        eval_time = 0
        adv_time = 0
        obs = self.env.reset()
        steps_taken = 0
        for _ in range(self.nsteps):
            # get information/instructions from model
            obs_tensor = torch.unsqueeze(torch.tensor(obs, dtype=torch.float).to(device),0)
            start = time.perf_counter()
            action_index, value, a_logit, se_logits = self.model.evaluate(obs_tensor)
            eval_time += time.perf_counter() - start
            stored_obs.append(obs)
            stored_actions.append(action_index)
            stored_vpreds.append(value)
            # tracking neg. log prob. of having gotten the sampled action
            stored_neglogprobs.append(a_logit)
            start = time.perf_counter()
            obs, reward, done = self.env.step(action_index)
            step_time += time.perf_counter() - start
            stored_rewards.append(reward)
            stored_se_logits.append(se_logits)
            if done:
                break
            steps_taken += 1
        # convert experience lists to torch tensors
        stored_obs = torch.tensor(stored_obs, dtype=torch.float).to(device)
        stored_rewards = torch.tensor(stored_rewards, dtype=torch.float).to(device)
        stored_vpreds = torch.tensor(stored_vpreds, dtype=torch.float).to(device)
        stored_neglogprobs = torch.tensor(stored_neglogprobs, dtype=torch.float).to(device)
        stored_se_logits = torch.tensor(stored_se_logits, dtype=torch.float).to(device)
        stored_actions = torch.tensor(stored_actions, dtype = torch.int).to(device)
        print(eval_time)
        print(step_time)
        # evaluate the final state
        _, last_value, _, _ = self.model.evaluate(obs_tensor)

        # use the stored values and rewards to calculate advantage estimates of the state-action pairs
        # (see Generalized Advantage Estimation paper)
        stored_advs = torch.zeros_like(stored_rewards)

        # Our adv. estimate is an exponentially-weighted average (EWA) over the n-step TD errors of the value of the state.
        # work backwards over the steps
        start = time.perf_counter()
        last_adv = 0
        for t in reversed(range(steps_taken)):
            if t == steps_taken - 1:
                nextvalue = last_value
            else:
                nextvalue = stored_vpreds[t + 1]
            current_value = stored_vpreds[t]

            # gamma is reward decay constant, lam is EWA "decay" constant

            # one-step TD error
            delta = stored_rewards[t] + self.gamma * nextvalue - current_value

            # the EWA of a step is equivalent to the decayed EWA of the next step + delta
            # (so you have work backwards from the last step)
            stored_advs[t] = last_adv = (last_adv * self.gamma * self.lam) + delta

        # for each step, its prior value plus its advantage estimate
        # is exactly the TD-lambda value estimate (by definition)
        # so stored_vtargets becomes the new target values for our Model's value function
        # (which is different from the action function but has shared hidden layers
        stored_vtargets = stored_advs + stored_vpreds
        adv_time = time.perf_counter() - start
        print(adv_time)
        return steps_taken, stored_obs, stored_rewards, stored_vpreds, stored_vtargets, stored_actions, stored_neglogprobs, stored_se_logits


def function_wrap(val):
    def f(_):
        return val
    return f

def learn(env, s_batch, total_timesteps, lr,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=4, epochs_per_batch=4, cliprange=0.1,
          save_interval=10):


    """
    VARIABLE NAMING CONVENTIONS
      prefix "sp_" means "steps per"
      prefix "n_" means "number of"
    """
    ob_space = env.observation_space
    ac_space = env.action_space
    total_timesteps = int(total_timesteps)

    n_batch = total_timesteps // s_batch
    s_minibatch = math.ceil(s_batch // nminibatches)

    model = Model(ob_space, ac_space, s_batch, vf_coef, lr)
    runner = Runner(env, model, s_batch, gamma, lam)
    loss_arr = []
    minibatches=0
    for i in range(n_batch):
        steps_taken, obs, reward, v_prev, v_target, action_index, a_logit_prev, se_logits = runner.run()
        print("1 run")
        inds = np.arange(steps_taken)
        for i in range(epochs_per_batch):
            print(loss_arr)
            # randomnly shuffle the steps into minibatches, for each epoch
            np.random.shuffle(inds)
            for start in range(0, steps_taken, s_minibatch):
                end = start + s_minibatch
                # the step indices for each minibatch
                mb_inds = inds[start:end]
                slices = (arr[mb_inds] for arr in (obs, v_prev, v_target, action_index, a_logit_prev, se_logits))
                start = time.perf_counter()
                loss_arr.append(model.train(*slices, cliprange))
                print("train_time: {}".format(time.perf_counter() - start))
                print("{} minibatch".format(minibatches))
                minibatches += 1
        print("1 update")

    return model
# converts the env's actions and observations into lists (representing matrices) if necessary
def obsConverter(obs):
    # doesn't do anything for this particular env,
    # only converted to Pytorch tensors when necessary as tensors are immutable
    return np.transpose(obs, (2,1,0))
class envWrapper():
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
    def reset(self):
        return obsConverter(self.env.reset())
    def step(self, action_index):
        obs, reward, done, _ = self.env.step(action_index)
        return obsConverter(obs), reward, done


def test():
    env = envWrapper(gym.make('Pong-v0'))
    model = learn(env, 5000, 1e5, 2e-4)
    total_reward = 0
    for i in range(300):
        obs = env.reset()
        while True:
            action_index, _ , _, _ = model.evaluate(torch.unsqueeze(torch.tensor(obs, dtype=torch.float).to(device),0))
            obs, reward, done = env.step(action_index)
            total_reward += reward
            if done:
                break

    total_reward_rand = 0
    for i in range(300):
        obs = env.reset()
        while True:
            obs, reward, done = env.step(env.env.action_space.sample())
            total_reward_rand += reward
            if done:
                break
    print("total_reward: {}".format(total_reward))
    print("total_reward_rand: {}".format(total_reward_rand))

test()

