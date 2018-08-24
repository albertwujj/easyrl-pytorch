import torch
import torch.nn as nn
from torch.distributions import
import numpy as np
import gym

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
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

shared = ConvNet().to(device)

class ActionFunction(nn.Module):
    def __init__(self, num_actions):
        super(ActionFunction, self).__init__()
        self.fc = nn.Linear(10, num_actions)

    def forward(self, x):
        return self.softmax(shared(x))

class ValueFunction(nn.Module):
    def __init__(self):
        super(ValueFunction, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(shared(x))

action_function = ActionFunction().to(device)
value_function = ValueFunction().to(device)



class Model():

    def __init__(self, obs_space, ac_space, nsteps, ent_coef, vf_coef, lr):
       self.obs_space, self.ac_space, self.nsteps, self.ent_coef, self.vf_coef \
           = obs_space, ac_space, nsteps, ent_coef, vf_coef
       self.parameters = list(shared.parameters()) + list(action_function.parameters()) + list(
           value_function.parameters())
       self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
       self.v_loss_weight = vf_coef

    def train(self, obs, v_prev, v_target, action_index, a_logit_prev, cliprange):

        # CALCULATE LOSS
        value = value_function(obs)
        v_clipped = torch.clamp(value - v_prev, -cliprange, cliprange) + v_prev
        v_loss = (value - v_target) ** 2
        v_loss_clipped = (v_clipped - v_target) ** 2
        # model will minimize the clipped or the non-clipped loss, whichever is greater
        v_loss_final = .5 * torch.mean(torch.max(v_loss, v_loss_clipped))

        a_logits = action_function(obs)
        # a_logit is the unscaled log of the actual action probability (a_prob), as no softmax has been applied.
        a_logit = a_logits[action_index]
        adv = v_target - v_prev

        # equivalent to dividing the actual a_prob by the previous actual a_prob
        ratio = torch.exp(a_logit - a_logit_prev)
        a_loss = - adv * ratio
        a_loss_clipped = - adv * torch.clamp(ratio, 1.0 - cliprange, cliprange)
        a_loss_final = .5 * torch.mean(torch.max(a_loss, a_loss_clipped))

        loss = a_loss_final + v_loss_final * self.v_loss_weight

        # GRADIENT DESCENT
        self.optimizer.zero_grad()
        loss.backwards() # compute gradients for all tensors where require_grad=True
        self.optimizer.step() # apply gradients



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
        stored_obs, stored_rewards, stored_actions, stored_vpreds, stored_neglogprobs = [], [], [], [], []

        obs = self.env.reset()
        steps_taken = 0
        for _ in range(self.nsteps):
            # get information/instructions from model
            action, value, neglogprob = self.model.step(obs)

            stored_obs.append(obs.copy())
            stored_actions.append(action)
            stored_vpreds.append(value)
            # tracking neg. log prob. of having gotten the sampled action
            stored_neglogprobs.append(neglogprob)
            obs, reward, done = self.env.step(action)
            stored_rewards.append(reward)
            steps_taken += 1
            if done:
                break

        # convert experience lists to torch tensors
        stored_obs = torch.tensor(stored_obs, dtype=self.obs.dtype, requires_grad=True)
        stored_rewards = torch.tensor(stored_rewards, dtype=np.float32, requires_grad=True)
        stored_actions = torch.tensor(stored_actions, dtype=int, requires_grad=True)
        stored_vpreds = torch.tensor(stored_vpreds, dtype=np.float32, requires_grad=True)
        stored_neglogprobs = torch.tensor(stored_neglogprobs, dtype=np.float32, requires_grad=True)

        # evaluate the final state
        last_value = self.model.value(obs)

        # use the stored values and rewards to calculate advantage estimates of the state-action pairs
        # (see Generalized Advantage Estimation paper)
        stored_advs = torch.zeros_like(stored_rewards)

        # Our adv. estimate is an exponentially-weighted average (EWA) over the n-step TD errors of the value of the state.

        # we work backwards over the steps to calculate the EWA, where
        # the bigger the n, the less the n-step error is weighted
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
            stored_advs[t] = (stored_advs[t+1] * self.gamma * self.lam) + delta

        # for each step, its prior value plus its advantage estimate
        # is exactly the TD-lambda value estimate (by definition)
        # so stored_vtargets becomes the new target values for our Model's value function
        # (which is different from the action function but has shared hidden layers
        stored_vtargets = stored_advs + stored_vpreds

        # TODO: Convert numpy arrays to pytorch tensors
        return steps_taken, stored_obs, stored_rewards, stored_vpreds, stored_vtargets, stored_actions, stored_neglogprobs


def function_wrap(val):
    def f(_):
        return val
    return f

def learn(env, s_batch, total_timesteps, ent_coef, lr,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=4, epochs_per_batch=4, cliprange=0.2,
          save_interval=10):


    """
    VARIABLE NAMING CONVENTIONS
      prefix "sp_" means "steps per"
      prefix "n_" means "number of"
    """
    ob_space = env.observation_space
    ac_space = env.action_space
    total_timesteps = int(total_timesteps)

    n_batch = total_timesteps / s_batch
    s_minibatch = s_batch / nminibatches

    model = Model(ob_space, ac_space, s_batch, ent_coef, vf_coef, lr)
    runner = Runner(env, model, s_batch, gamma, lam)

    for i in n_batch:
        steps_taken, obs, reward, v_prev, v_target, action_index, a_logit_prev = runner.Run()
        inds = np.arange(s_batch)
        for i in epochs_per_batch:
            # randomnly shuffle the steps into minibatches, for each epoch
            np.random.shuffle(inds)
            for start in range(0, s_batch, s_minibatch):
                end = start + s_minibatch
                # the step indices for each minibatch
                mb_inds = inds[start:end]
                slices = (arr[mb_inds] for arr in (obs, v_prev, v_target, action_index, a_logit_prev))
                model.train(*slices, cliprange)

def test():
    env = gym.make('CartPole-v0')
    learn(env, 4096, int(5e7), 0, 2e-4)
test()