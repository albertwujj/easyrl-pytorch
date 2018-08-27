import torch
import torch.nn as nn
import torch.nn.modules.loss as loss
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
class NeuralNet(nn.Module):
    def __init__(self, obs_length):
        """
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
        """
        # Can be changed freely, e.g. to a CNN architecture
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(obs_length, 20)
        self.layer2 = nn.Linear(20, 10)
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out)
        return out



class ActionFunction(nn.Module):
    def __init__(self, shared, num_actions):
        super(ActionFunction, self).__init__()
        self.shared = shared
        self.fc = nn.Linear(10, num_actions)

    def forward(self, x):
        return self.fc(self.shared(x))

class ValueFunction(nn.Module):
    def __init__(self, shared):
        super(ValueFunction, self).__init__()
        self.shared = shared
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(self.shared(x))

class PPOLoss(loss._Loss):
    def __init__(self, action_function, value_function, size_average=None, reduce=None, reduction='sum'):
        super(PPOLoss, self).__init__(size_average, reduce, reduction)
        self.action_function = action_function
        self.value_function = value_function

    def forward(self, obs, v_prev, v_target, action_index, a_logit_prev, vf_coef, cliprange):
        # CALCULATE LOSS
        value = self.value_function(obs)
        v_clipped = torch.clamp(value - v_prev, -cliprange, cliprange) + v_prev
        v_loss = (value - v_target) ** 2
        v_loss_clipped = (v_clipped - v_target) ** 2
        # model will minimize the clipped or the non-clipped loss, whichever is greater
        v_loss_final = .5 * torch.mean(torch.max(v_loss, v_loss_clipped))

        a_logits = self.action_function(obs)
        # a_logit is the unscaled log of the actual action probability (a_prob), as no softmax has been applied.
        selected_a_logit = a_logits[:, [i for i in action_index]]
        adv = v_target - v_prev

        # equivalent to dividing the actual (after softmax) a_prob by the previous actual a_prob
        ratio = torch.exp(selected_a_logit - a_logit_prev)
        a_loss = - adv * ratio
        a_loss_clipped = - adv * torch.clamp(ratio, 1.0 - cliprange, cliprange)
        a_loss_final = .5 * torch.mean(torch.max(a_loss, a_loss_clipped))

        loss = a_loss_final + v_loss_final * vf_coef
        return loss

class Model():

    def __init__(self, obs_space, ac_space, nsteps, lr):
       shared_layers = NeuralNet(obs_space).to(device)
       self.action_function = ActionFunction(shared_layers, ac_space).to(device)
       self.value_function = ValueFunction(shared_layers).to(device)
       self.obs_space, self.ac_space, self.nsteps = obs_space, ac_space, nsteps
       self.parameters = list(shared_layers.parameters()) + list(self.action_function.parameters()) + list(
           self.value_function.parameters())
       self.optimizer = torch.optim.Adam(self.parameters, lr=lr)

    # each tensor can actually represent more than 1 step. First dimension is step #
    def train(self, obs, v_prev, v_target, action_index, a_logit_prev, vf_coef, cliprange):
        criterion = PPOLoss(self.action_function, self.value_function)
        loss = criterion(obs, v_prev, v_target, action_index, a_logit_prev, vf_coef, cliprange)
        # GRADIENT DESCENT
        self.optimizer.zero_grad()
        loss.backward() # compute gradients
        self.optimizer.step() # apply gradients

    def evaluate(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float)
        a_logits = self.action_function(obs_tensor)
        a_logit, action_index = a_logits.max(0)
        value = self.value_function(obs_tensor)
        return action_index.item(), value, a_logit

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
            action_index, value, a_logit = self.model.evaluate(obs)

            stored_obs.append(obs)
            stored_actions.append(action_index)
            stored_vpreds.append(value)
            # tracking neg. log prob. of having gotten the sampled action
            stored_neglogprobs.append(a_logit)
            obs, reward, done = self.env.step(action_index)
            stored_rewards.append(reward)
            if done:
                break
            steps_taken += 1

        # convert experience lists to torch tensors
        stored_obs = torch.tensor(stored_obs, dtype=torch.float)
        stored_rewards = torch.tensor(stored_rewards, dtype=torch.float)
        stored_vpreds = torch.tensor(stored_vpreds, dtype=torch.float)
        stored_neglogprobs = torch.tensor(stored_neglogprobs, dtype=torch.float)
        stored_actions = torch.tensor(stored_actions, dtype = torch.int)

        # evaluate the final state
        _, last_value, _ = self.model.evaluate(obs)

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

def learn(env, s_batch, total_timesteps, lr,
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

    n_batch = total_timesteps // s_batch
    s_minibatch = s_batch // nminibatches

    model = Model(ob_space, ac_space, s_batch, lr)
    runner = Runner(env, model, s_batch, gamma, lam)

    for i in range(n_batch):
        steps_taken, obs, reward, v_prev, v_target, action_index, a_logit_prev = runner.run()
        inds = np.arange(steps_taken)
        for i in range(epochs_per_batch):
            # randomnly shuffle the steps into minibatches, for each epoch
            np.random.shuffle(inds)
            for start in range(0, steps_taken, s_minibatch):
                end = start + s_minibatch
                # the step indices for each minibatch
                mb_inds = inds[start:end]
                slices = (arr[mb_inds] for arr in (obs, v_prev, v_target, action_index, a_logit_prev))
                model.train(*slices, vf_coef, cliprange)


# converts the env's actions and observations into lists (representing matrices) if necessary
def obsToList(obs):
    # doesn't do anything for this particular env,
    # only converted to Pytorch tensors when necessary as tensors are immutable
    return obs
class envWrapper():
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
    def reset(self):
        return obsToList(self.env.reset())
    def step(self, action_index):
        obs, reward, done, _ = self.env.step(action_index)
        return obsToList(obs), reward, done


def test():
    env = envWrapper(gym.make('CartPole-v0'))
    learn(env, 4096, 1e5, 2e-4)

test()

