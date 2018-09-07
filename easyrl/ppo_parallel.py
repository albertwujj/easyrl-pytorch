import torch
import torch.nn as nn
import numpy as np
import gym
import retro
import math
import time


from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from sonic_util import make_envs

from atari_wrappers import WarpFrame, FrameStack

import logging

logging.basicConfig(filename="losses.log", level=logging.DEBUG)


""" A readable, thoroughly commented implementation of PPO
"""

# TODO: Add parallel env support
# TODO: Keep track of action probs more efficiently (stop using sum_exp_logit)
# TODO: Reset env after done to continue adding steps to batch until hitting batch_size
# TODO: Calculate the # of input channels for convolutional layers automatically
# TODO: Add RNN support

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() and False else 'cpu')


# The neural network outputting both action and value
class ConvNet(nn.Module):
    def __init__(self, obs_length, num_actions):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 5, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(4840, 10)
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
    z0 = torch.sum(ea0, -1, keepdim=True)
    p0 = ea0 / z0
    return torch.sum(p0 * (torch.log(z0) - a0), -1)

# Will take experience tuples, calculate PPO loss and train on it,
# to predict action dist. and value for each obs
class Model():
    def __init__(self, obs_space, ac_space, nsteps, vf_coef, lr):
        self.nn = ConvNet(obs_space, ac_space).to(device)
        self.obs_space, self.ac_space, self.nsteps, self.vf_coef \
            = obs_space, ac_space, nsteps, vf_coef
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=lr)
        self.vf_coef = vf_coef

    # each tensor can actually represent more than 1 step. First dimension is step #
    def train(self, obs, v_prev, v_target, action_index, a_logit_prev, sum_exp_logits_prev, cliprange):
        v_prev = torch.tensor(v_prev, dtype=torch.float).to(device)
        v_target = torch.tensor(v_target, dtype=torch.float).to(device)
        obs = torch.tensor(obs, dtype=torch.float).to(device)
        a_logit_prev = torch.tensor(a_logit_prev, dtype=torch.float).to(device)
        sum_exp_logits_prev = torch.tensor(sum_exp_logits_prev, dtype=torch.float).to(device)
        action_index = torch.tensor(action_index, dtype=torch.int).to(device)

        value, a_logits = self.nn(obs)
        # VALUE LOSS
        v_clipped = torch.clamp(value - v_prev, -cliprange, cliprange) + v_prev
        v_loss = (value - v_target) ** 2
        v_loss_clipped = (v_clipped - v_target) ** 2
        # model will minimize the clipped or the non-clipped loss, whichever is greater
        v_loss_final = .5 * torch.mean(torch.max(v_loss, v_loss_clipped))

        # ACTION LOSS (PPO LOSS)
        sum_exp_logits = torch.sum(torch.exp(a_logits), -1)
        entropy = torch.mean(calculateEntropy(a_logits))
        # the unscaled log of the actual action probability, as no softmax has been applied.
        selected_a_logit = a_logits[np.arange(a_logits.shape[0]), [i for i in action_index]]
        adv = v_target - v_prev
        # equivalent to dividing the predicted prob. by the previous predicted prob.
        ratio = torch.exp(selected_a_logit - a_logit_prev) / (sum_exp_logits * sum_exp_logits_prev)
        a_loss = - adv * ratio
        a_loss_clipped = - adv * torch.clamp(ratio, 1.0 - cliprange, cliprange)
        a_loss_final = .5 * torch.mean(torch.max(a_loss, a_loss_clipped))

        loss = a_loss_final + v_loss_final * self.vf_coef
        print(
            "a_loss {}, v_loss {}, entropy {} loss {}".format(a_loss_final.item(), v_loss_final.item(), entropy.item(),
                                                              loss.item()))

        # GRADIENT DESCENT
        self.optimizer.zero_grad()
        loss.backward()  # compute gradients
        self.optimizer.step()  # apply gradients

        return v_loss_final.item(), a_loss_final.item(), loss.item(), entropy.item()

    def evaluate(self, obs_tensor):
        """
        first dimension of input/output is # of observations
        """
        value, a_logits = self.nn(obs_tensor)
        value = value
        a_logits = a_logits
        sum_exp_logits = torch.sum(torch.exp(a_logits), -1)
        a_logit, action_index = a_logits.max(-1)
        return action_index, value, a_logit, sum_exp_logits


# This class will use the model to take actions in the environment.
# It will return a tuple of experience (observations[], actions[], rewards[]), along with the advantages it calculates
# for each step
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

        # tracking data points for each step (specifically, each observation)
        stored_obs, stored_rewards, stored_actions, stored_vpreds, stored_a_logits, stored_sum_exp_logits = [], [], [], [], [], []
        stored_dones = []

        eval_time = 0
        obs = self.env.reset() # first obs (so we actually have (nsteps+1) obs)

        done = np.zeros((self.env.num_envs), dtype=bool)
        reward = np.zeros((self.env.num_envs))
        for _ in range(self.nsteps):

            start = time.perf_counter()
            obs_tensor = torch.tensor(obs, dtype=torch.float).to(device)
            action_index, value, a_logit, se_logits = self.model.evaluate(obs_tensor)
            eval_time += time.perf_counter() - start

            # if env contains multiple envs,
            # first dimension of obs, action_index, etc. is # envs
            stored_obs.append(obs)
            stored_actions.append(action_index)
            stored_vpreds.append(value)
            stored_a_logits.append(a_logit)
            if reward is None:
                reward = np.zeros((self.env.num_envs))
            stored_rewards.append(reward)
            stored_dones.append(done)
            stored_sum_exp_logits.append(se_logits)

            obs, reward, done = self.env.step(action_index)
            # experience is not recorded for the final step




        _, last_value, _, _ = self.model.evaluate(obs_tensor) # evaluate the final state
        # convert experience lists to numpy arrays
        # (Do not convert to Pytorch tensors until feeding into network,
        # (as the backprop computation graph starts being built from the first tensor)
        stored_obs = np.asarray(stored_obs, dtype=np.float32)
        print("hey")
        print(stored_dones)
        stored_rewards = np.asarray(stored_rewards, dtype=np.float32)
        stored_a_logits = np.asarray(stored_a_logits, dtype=np.float32)
        stored_sum_exp_logits = np.asarray(stored_sum_exp_logits, dtype=np.float32)
        stored_actions = np.asarray(stored_actions, dtype=np.float32)
        stored_vpreds = np.asarray(stored_vpreds, dtype=np.float32)
        stored_dones = np.asarray(stored_dones, dtype=np.bool)


        # use the stored values and rewards to calculate advantage estimates of the state-action pairs
        # (see Generalized Advantage Estimation paper)
        stored_advs = np.zeros_like(stored_rewards)

        # Our adv. estimate is an exponentially-weighted average (EWA) over the n-step TD errors of the value of the state.
        last_adv = 0
        for t in reversed(range(self.nsteps)):
            # we will technically have (nsteps+1) steps,
            # but data for the final step is not returned (as we cannot calculate a value target/adv for the last step)
            # we just use it to calculate values/adv for the previous steps
            if t == self.nsteps - 1:
                nextvalue = last_value
                nextnotdones = 1.0 - done

            else:
                nextvalue = stored_vpreds[t + 1]
                # will contain 0 for any envs that are done at the next step,
                # used to ensure the next-value used for curr step is 0
                nextnotdones = 1.0 - stored_dones[t+1]
            current_value = stored_vpreds[t]

            # gamma is reward decay constant, lam is EWA "decay" constant
            delta = stored_rewards[t] + self.gamma * nextvalue * nextnotdones - current_value # one-step TD error

            # the EWA of a step is equivalent to the decayed EWA of the next step + delta
            # (so you have work backwards from the last step)
            stored_advs[t] = last_adv = (last_adv * self.gamma * self.lam * nextnotdones) + delta

        # for each step, its prior value plus its advantage estimate
        # is exactly the TD-lambda value estimate (by definition)
        # so stored_vtargets becomes the new target values for our Model's value function
        stored_vtargets = stored_advs + stored_vpreds

        arrs = (stored_obs, stored_rewards, stored_vpreds, stored_vtargets, stored_actions, stored_a_logits, stored_sum_exp_logits)
        return map(swap01_flatten, arrs)


# used for multiple envs
def swap01_flatten(arr):
    """
      This function will swap axis 0 (steps) and 1 (envs),
      so the steps will be grouped by environment after flattening
    """
    arr = arr.swapaxes(0, 1)
    shape = arr.shape
    new_shape = (shape[0] * shape[1], *shape[2:]) # flatten 0 and 1
    return arr.reshape(new_shape)


def function_wrap(val):
    def f(_):
        return val

    return f

# this function will call the runner for a certain number of steps,
# arrange the returned experience into batches and feed it into the model,
# then repeat
def learn(env, s_batch, total_timesteps, lr,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=1, nminibatches=4, epochs_per_batch=4, cliprange=0.1,
          save_interval=10):
    """
    VARIABLE NAMING CONVENTIONS
      prefix "s_" means "steps per"
      prefix "n_" means "number of"
    """
    ob_space = env.observation_space
    ac_space = env.action_space
    total_timesteps = int(total_timesteps)

    n_batch = total_timesteps // s_batch


    model = Model(ob_space, ac_space, s_batch, vf_coef, lr)
    runner = Runner(env, model, s_batch, gamma, lam)


    for batch in range(n_batch):
        loss_arr = []
        obs, reward, v_prev, v_target, action_index, a_logit_prev, se_logits = runner.run() # collect a batch of data
        inds = np.arange(s_batch)
        for epoch in range(epochs_per_batch):
            np.random.shuffle(inds) # randomnly shuffle the steps into minibatches, for each epoch
            minibatches = 0
            s_minibatch = math.ceil(s_batch // nminibatches)
            for start in range(0, s_batch, s_minibatch):
                end = start + s_minibatch
                mb_inds = inds[start:end]  # the step indices for each minibatch
                slices = (arr[mb_inds] for arr in (obs, v_prev, v_target, action_index, a_logit_prev, se_logits))
                start = time.perf_counter()
                loss_arr.append(model.train(*slices, cliprange))
                #print("train_time: {}".format(time.perf_counter() - start))
                minibatches += 1
                print("{}, {}, {} b e mb".format(batch + 1, epoch + 1, minibatches))
        if batch != 0 and batch % log_interval == 0:
            logging.debug("Batch {}, losses (v,a,total,entropy)= {}".format(batch, loss_arr))

    return model


def obsConverter(obs):
    # changes observation's channel dimension to be it's 2nd
    return np.transpose(obs, (0,3, 2, 1))

# change this to work w/ diff envs
class envWrapper():
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.num_envs = env.num_envs

    def reset(self):
        return obsConverter(self.env.reset())

    def step(self, action_index):
        obs, reward, done, _ = self.env.step(action_index) # output is numpy array, we want list
        return obsConverter(obs), reward, done


def test():
    env = envWrapper(SubprocVecEnv(make_envs(num=2)))
    model = learn(env, 3000, 3e4, 2e-4)
    total_reward = 0
    for i in range(30):
        obs = env.reset()
        while True:
            action_index, _, _, _ = model.evaluate(torch.unsqueeze(torch.tensor(obs, dtype=torch.float).to(device), 0)) # need to unsqueeze eval output
            obs, reward, done = env.step(action_index)
            print(action_index)
            total_reward += reward
            if done:
                break
        print("{} testgames done".format(i))
    total_reward_rand = 0
    for i in range(30):
        obs = env.reset()
        while True:
            obs, reward, done = env.step(env.env.action_space.sample())
            total_reward_rand += reward
            if done:
                break
        print("{} testgames done".format(i))
    print("total_reward: {}".format(total_reward))
    print("total_reward_rand: {}".format(total_reward_rand))


test()