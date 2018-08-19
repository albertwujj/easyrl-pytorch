import torch
import numpy as np

# This class will follow output from the model to take actions in the environment.
# It will return the tuples of experience (observation, action, reward) it receives
class Runner(object):
    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        # initial observations

        # parameters for calculating the advantage of a state-action pair
        self.gamma = gamma
        self.lam = lam

        self.nsteps = nsteps

    def run(self):
        # the experience to return (to eventually send to the model)
        stored_obs, stored_rewards, stored_actions, stored_values, stored_neglogprobs = [], [], [], [], []

        obs = self.env.reset()
        for _ in range(self.nsteps):
            # get information/instructions from model
            action, value, neglogprob = self.model.step(obs)

            stored_obs.append(obs.copy())
            stored_actions.append(action)
            stored_values.append(value)
            # tracking neg. log prob. of having gotten the sampled action
            stored_neglogprobs.append(neglogprob)
            obs, reward, done = self.env.step(action)
            stored_rewards.append(reward)
            if done:
                break

        # convert experience lists to numpy arrays
        stored_obs, stored_rewards, stored_actions, stored_values, stored_neglogprobs = [], [], [], [], [],
        stored_obs = np.asarray(stored_obs, dtype=self.obs.dtype)
        stored_rewards = np.asarray(stored_rewards, dtype=np.float32)
        stored_actions = np.asarray(stored_actions)
        stored_values = np.asarray(stored_values, dtype=np.float32)
        stored_neglogprobs = np.asarray(stored_neglogprobs, dtype=np.float32)

        # evaluate the final state
        last_value = self.model.value(obs)

        # use the stored values and rewards to calculate advantages of the state-action pairs
        # (see Generalized Advantage Estimation paper)
        stored_advs = np.zeros_like(stored_rewards)

        # essentially, we want to take an average over the n-step TD errors of the value of the state,
        # subtracted by the current value of the state

        # we work backwards over the steps to calculate the exponentially-weighted average (EWA), where
        # the larger n n-step errors are worth less
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextvalue = last_value
            else:
                nextvalue = stored_values[t+1]
            current_value = stored_values[t]

            # gamma is reward decay constant, lam is EWA "decay" constant

            # one-step TD error
            delta = stored_rewards[t] + self.gamma * nextvalue - current_value

            # finding the EWA of the TD errors is equivalent to setting each step's advantage to
            # the advantage of the next step times gamma times lambda, plus its one-step TD error
            stored_advs[t] = (stored_advs[t+1] * self.gamma * self.lam) + delta