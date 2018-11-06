import torch
import torch.nn as nn
import numpy as np
import gym
import retro
import math
import time
import sys
import random

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import sonic_util as sonic
from easyrl.neural_nets.conv import conv

from baselines.ppo2 import ppo2

from atari_wrappers import WarpFrame, FrameStack

import logging

logging.basicConfig(filename="losses.log", level=logging.DEBUG)


""" A readable, thoroughly commented implementation of PPO
    (everything in one file for tutorial purposes)
"""



# TODO: Investigate why exploding value loss w/o maxpool layers
# TODO: Test performance against OpenAI Baselines
# TODO: Log losses, etc in same manner as OpenAI Baselines
# TODO: Keep track of action probs more efficiently (stop using sum_exp_logit)
# TODO: Add RNN support

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() and False else 'cpu')


# The neural network outputting both action and value
class ConvNet(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super(ConvNet, self).__init__()

        # shape of 2D input (cutting out batch and channel dims)
        shape0 = (obs_shape[2], obs_shape[3])


        c0 = obs_shape[1]  # num channels of input
        c1 = 32  # num of output channels of first layer
        c2 = 64
        c3 = 64

        fc_out = 512  # a choice


        self.layer1, shape1 = conv(shape0, c0, c1, kernel_size=3, stride=4)
        self.layer2, shape2 = conv(shape1, c1, c2, kernel_size=4, stride=2)
        self.layer3, shape3 = conv(shape2, c2, c3, kernel_size=3, stride=1)

        fc_in = 3136
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(fc_in, fc_out))
        self.fcAction = nn.Linear(fc_out, num_actions)
        self.fcValue = nn.Linear(fc_out, 1)

        """   
        self.layer1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(4840, c4)
        self.fcAction = nn.Linear(c4, num_actions)
        self.fcValue = nn.Linear(c4, 1)
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x / 255
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return self.fcValue(out), self.fcAction(out)


# Will take steps of experience and calculate loss to minimize as per PPO
# Predicts action distribution and value for each obs
class Model():
    def __init__(self, obs_space, ac_space, nsteps, vf_coef, lr):
        self.nn = ConvNet(obs_space, ac_space).to(device)
        self.obs_space, self.ac_space, self.nsteps, self.vf_coef \
            = obs_space, ac_space, nsteps, vf_coef
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=lr)
        self.vf_coef = vf_coef

    @staticmethod
    def calculate_value_loss(value, v_prev, cliprange, v_target):
        v_clipped = torch.clamp(value - v_prev, -cliprange, cliprange) + v_prev
        v_loss = (value - v_target) ** 2
        v_loss_clipped = (v_clipped - v_target) ** 2
        # model will minimize the clipped or the non-clipped loss, whichever is greater
        return .5 * torch.mean(torch.max(v_loss, v_loss_clipped))

    @staticmethod
    def calculate_action_loss(a_logits, action_index, a_logit_prev, cliprange, sum_exp_logits_prev, adv):
        sum_exp_logits = torch.sum(torch.exp(a_logits), -1)
        # the unscaled log of the actual action dist. probabilities, as no softmax has been applied.
        selected_a_logit = a_logits[np.arange(a_logits.shape[0]), [i for i in action_index]]
        # equivalent to dividing the predicted prob. by the previous predicted prob.
        ratio = torch.exp(selected_a_logit - a_logit_prev)  * sum_exp_logits_prev / sum_exp_logits
        a_loss = - adv * ratio
        a_loss_clipped = - adv * torch.clamp(ratio, 1.0 - cliprange, cliprange)
        approxkl = .5 * torch.mean((selected_a_logit - a_logit_prev) ** 2)
        return .5 * torch.mean(torch.max(a_loss, a_loss_clipped)), approxkl

    @staticmethod
    def calculateEntropy(logits):
        a0 = logits - torch.max(logits, -1, keepdim=True)[0]
        ea0 = torch.exp(a0)
        z0 = torch.sum(ea0, -1, keepdim=True)
        p0 = ea0 / z0
        return torch.sum(p0 * (torch.log(z0) - a0), -1)

    # each tensor can actually represent more than 1 step. First dimension is step #
    def train(self, obs, v_prev, v_target, action_index, a_logit_prev, sum_exp_logits_prev, cliprange):

        # convert data from Runner to tensors
        v_prev = torch.tensor(v_prev, dtype=torch.float).to(device)
        v_target = torch.tensor(v_target, dtype=torch.float).to(device)
        obs = torch.tensor(obs, dtype=torch.float).to(device)
        a_logit_prev = torch.tensor(a_logit_prev, dtype=torch.float).to(device)
        sum_exp_logits_prev = torch.tensor(sum_exp_logits_prev, dtype=torch.float).to(device)
        action_index = torch.tensor(action_index, dtype=torch.int).to(device)
        adv = v_target - v_prev

        adv = (adv - adv.mean()) / (adv.std() + 1e-8) # normalize advantages
        value, a_logits = self.nn(obs)

        v_loss = Model.calculate_value_loss(value, v_prev, cliprange, v_target)
        a_loss, approxkl = Model.calculate_action_loss(a_logits, action_index, a_logit_prev, cliprange, sum_exp_logits_prev, adv)
        entropy = torch.mean(Model.calculateEntropy(a_logits))

        loss = a_loss + v_loss * self.vf_coef
        print(
            "a_loss {}, v_loss {}, entropy {} loss {}".format(a_loss.item(), v_loss.item(), entropy.item(),
                                                              loss.item()))

        # GRADIENT DESCENT
        self.optimizer.zero_grad()
        loss.backward()  # compute gradients
        self.optimizer.step()  # apply gradients

        return v_loss.item(), a_loss.item(), approxkl.item()

    def eval_and_sample(self, obs_tensor):
        """
        first dimension of value, a_logits is # of observations
        """
        value, a_logits = self.nn(obs_tensor) # tensor output from NN
        # squeeze value from shape (num_obs, 1) to (num_obs)
        value = torch.squeeze(value, -1).detach().numpy()
        a_logits = a_logits.detach().numpy()

        sum_exp_logits = np.sum(np.exp(a_logits), -1)
        a_probs = np.exp(a_logits) / np.expand_dims(sum_exp_logits, -1)  # apply softmax
        sample_row = lambda row: np.random.choice(row.shape[0], p=row)
        a_i = np.apply_along_axis(sample_row, 1, a_probs)  # the row to sample from is the action dist (a_probs)
        a_logit = a_logits[np.arange(a_logits.shape[0]), a_i]

        return a_i, value, a_logit, sum_exp_logits

# This class will take actions in the environment, based on output from the model.
# It will return a tuple of experience (observations[], actions[], rewards[]), along with the advantages it calculates
# for each step
class Runner(object):
    def __init__(self, m_env, model, nsteps, gamma, lam):
        # NOTE: m_env is actually a collection of envs.
        # m_env's inputs/outputs should be a numpy array where the 1st dimension is num_envs
        self.m_env = m_env
        self.model = model
        # initial observations

        # parameters for calculating the advantage of a state-action pair
        self.gamma = gamma
        self.lam = lam

        self.nsteps = nsteps

    def run(self):

        # tracking data points for each step (specifically, each observation)
        # (2+D arrays of shape (num_steps, num_envs) + whatever extra dimensions obs has
        stored_obs, stored_rewards, stored_actions, stored_vpreds, stored_a_logits, stored_sum_exp_logits = [], [], [], [], [], []
        stored_dones = []

        eval_time = 0

        # ob is a 1D array of observations for each env
        ob = self.m_env.reset() # first obs (so we actually have (nsteps+1) obs)

        done = np.zeros((self.m_env.num_envs), dtype=bool)
        reward = np.zeros((self.m_env.num_envs))
        for _ in range(self.nsteps):

            start = time.perf_counter()
            obs_tensor = torch.tensor(ob, dtype=torch.float).to(device)
            action_index, value, a_logit, se_logits = self.model.eval_and_sample(obs_tensor)

            eval_time += time.perf_counter() - start

            # first dimension of obs, action_index, etc. is num_envs
            stored_obs.append(ob)

            stored_actions.append(action_index)
            stored_vpreds.append(value)
            stored_a_logits.append(a_logit)
            if reward is None:
                reward = np.zeros((self.m_env.num_envs))
            stored_rewards.append(reward)
            stored_dones.append(done)
            stored_sum_exp_logits.append(se_logits)

            ob, reward, done = self.m_env.step(action_index)
            # experience is not recorded for the final step

        # convert experience lists to numpy arrays
        stored_obs = np.asarray(stored_obs, dtype=np.float32)
        stored_rewards = np.asarray(stored_rewards, dtype=np.float32)
        stored_a_logits = np.asarray(stored_a_logits, dtype=np.float32)
        stored_sum_exp_logits = np.asarray(stored_sum_exp_logits, dtype=np.float32)
        stored_actions = np.asarray(stored_actions, dtype=np.float32)
        stored_vpreds = np.asarray(stored_vpreds, dtype=np.float32)
        stored_dones = np.asarray(stored_dones, dtype=np.bool)

        # use the values (from model) and rewards (from env) to calculate advantage estimates
        # and new value targets for the state-action pairs
        stored_advs = np.zeros_like(stored_rewards)
        _, last_value, _, _ = self.model.eval_and_sample(obs_tensor)  # value of the final step

        # Our adv. estimate is an exponentially-weighted average (EWA) over the n-step TD errors of the value of the state.
        # (see Generalized Advantage Estimation paper)
        last_adv = np.zeros((self.m_env.num_envs))
        for t in reversed(range(self.nsteps)):
            # we will technically have (nsteps+1) steps,
            # but will not return data for the final step (as we cannot calculate a value target/adv for the last step)
            # we just use it to calculate values/adv for the previous steps
            if t == self.nsteps - 1:
                nextvalue = last_value
                nextnotdones = 1.0 - done
            else:
                nextvalue = stored_vpreds[t + 1]
                # will contain 0 for any envs that are done at step t+1
                nextnotdones = 1.0 - stored_dones[t+1]

            current_value = stored_vpreds[t]

            # gamma is reward decay constant, lam is EWA "decay" constant
            delta = stored_rewards[t] + self.gamma * nextvalue * nextnotdones - current_value # one-step TD error

            # the EWA of a step is equivalent to delta + the decayed EWA of the next step
            # (so you have work backwards from the last step)
            stored_advs[t] = last_adv = (last_adv * self.gamma * self.lam * nextnotdones) + delta

        # for each step, its prior value plus its GAE advantage estimate
        # is exactly the TD-lambda value estimate (by definition)
        # so stored_vtargets are the new target values for our Model's value function
        stored_vtargets = stored_advs + stored_vpreds

        arrs = (stored_obs, stored_rewards, stored_vpreds, stored_vtargets, stored_actions, stored_a_logits, stored_sum_exp_logits)
        return map(swap01_flatten, arrs)


def swap01_flatten(arr):
    """
      This function will swap axis 0 (steps) and 1 (envs),
      so the steps will be grouped by environment
      after flattening
    """
    arr = arr.swapaxes(0, 1)
    shape = arr.shape
    new_shape = (shape[0] * shape[1], *shape[2:]) # flatten 0 and 1
    return arr.reshape(new_shape)


# this function will call the runner for s_batch steps,
# arrange the returned experience into minibatches and feed it into the model,
# repeat until total_timesteps
def learn(*, env, s_env, total_timesteps, lr,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=1, nminibatches=8, epochs_per_batch=4, cliprange=0.1,
          save_interval=10):
    """
    VARIABLE NAMING CONVENTIONS
      prefix "s_" means "steps per"
      prefix "n_" means "number of"
    """
    ob_space = env.observation_space
    ac_space = env.action_space
    total_timesteps = int(total_timesteps)

    s_batch = s_env * env.num_envs
    n_batch = total_timesteps // s_batch

    model = Model(ob_space, ac_space, s_batch, vf_coef, lr)
    runner = Runner(env, model, s_env, gamma, lam)


    for batch in range(n_batch):
        loss_arr = []
        obs, reward, v_prev, v_target, action_index, a_logit_prev, se_logits = runner.run() # collect a batch of data
        inds = np.arange(s_batch)
        for epoch in range(epochs_per_batch):

            np.random.shuffle(inds) # randomnly shuffle the steps into minibatches, for each epoch
            minibatches = 0
            s_minibatch = math.ceil(s_batch // nminibatches)
            assert s_minibatch > 0
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
    # changes our env's observation's channel dimension to be it's 2nd
    # (required by pytorch Conv layer)
    return np.transpose(obs, (0, 3, 2, 1))

# change this to work w/ your custom envs
# 1st dimension of input/output should be num_envs, even if num_envs = 1
class envWrapper():
    def __init__(self, env):
        self.env = env
        example_obs = np.expand_dims(np.zeros(env.observation_space.shape),0)
        self.observation_space = obsConverter(example_obs).shape
        self.action_space = env.action_space.n
        self.num_envs = env.num_envs

    def reset(self):
        return obsConverter(self.env.reset())

    def step(self, action_index):
        obs, reward, done, _ = self.env.step(action_index)
        return obsConverter(obs), reward, done




def test():
    num_envs = 8
    s_env = 2048
    total_timesteps= 4e4


    env = envWrapper(SubprocVecEnv(sonic.make_envs(num=num_envs)))
    model = learn(env=env, s_env=s_env, total_timesteps=total_timesteps, lr=3e-4, lam=0.95,
                  gamma=0.99)
    total_reward = 0.0
    for i in range(30):
        obs = env.reset()
        while True:
            action_index, _, _, _ = model.eval_and_sample(
                torch.tensor(obs, dtype=torch.float).to(device))  # need to unsqueeze eval output
            obs, reward, done = env.step(action_index)
            total_reward += np.sum(reward)
            if done.all():
                break
        print("{} testgames done".format(i + 1))

    env_openAI = SubprocVecEnv(sonic.make_envs(num=num_envs))
    model_openAI = ppo2.learn(network="cnn", env=env_openAI, total_timesteps=total_timesteps, nsteps=s_env,
                              nminibatches=8,
                              lam=0.95,
                              gamma=0.99,
                              noptepochs=3,
                              log_interval=1,
                              ent_coef=0.01,
                              lr=lambda _: 3e-4,
                              cliprange=lambda _: 0.1,
                              save_interval=5)
    total_reward_openAI = 0.0
    for i in range(30):
        obs = env_openAI.reset()
        done = np.zeros((env.num_envs), dtype=bool)
        while True:
            action_index, values,_, neglogpacs = model_openAI.step(obs, S=None, M=done)
            obs, reward, done = env_openAI.step(action_index)
            total_reward_openAI += np.sum(reward)
            if done.all():
                break
        print("{} testgames done".format(i + 1))

    total_reward_rand = 0.0
    for i in range(30):
        obs = env.reset()
        while True:
            obs, reward, done = env.step([env.env.action_space.sample() for i in range(env.num_envs)])
            total_reward_rand += np.sum(reward)
            if done.all():
                break
        print("{} testgames done".format(i + 1))
        print("")

    print("total_reward_openAI: {}".format(total_reward_openAI))
    print("total_reward: {}".format(total_reward))
    print("total_reward_rand: {}".format(total_reward_rand))


test()