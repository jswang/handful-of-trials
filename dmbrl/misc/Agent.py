from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from dotmap import DotMap

import time


class Agent:
    """An general class for RL agents.
    """
    def __init__(self, params):
        """Initializes an agent.

        Arguments:
            params: (DotMap) A DotMap of agent parameters.
                .env: (OpenAI gym environment) The environment for this agent.
                .noisy_actions: (bool) Indicates whether random Gaussian noise will
                    be added to the actions of this agent.
                .noise_stddev: (float) The standard deviation to be used for the
                    action noise if params.noisy_actions is True.
        """
        self.env = params.env

        self.noise_stddev = params.noise_stddev if params.get("noisy_actions", False) else None

        if isinstance(self.env, DotMap):
            raise ValueError("Environment must be provided to the agent at initialization.")
        if (not isinstance(self.noise_stddev, float)) and params.get("noisy_actions", False):
            raise ValueError("Must provide standard deviation for noise for noisy actions.")

        if self.noise_stddev is not None:
            self.dU = self.env.action_space.shape[0]

    def sample(self, horizon, policy, record_fname=None):
        """Samples a rollout from the agent.

        Arguments:
            horizon: (int) The length of the rollout to generate from the agent.
            policy: (policy) The policy that the agent will use for actions.
            record_fname: (str/None) The name of the file to which a recording of the rollout
                will be saved. If None, the rollout will not be recorded.

        Returns: (dict) A dictionary containing data from the rollout.
            The keys of the dictionary are 'obs', 'ac', and 'reward_sum'.
        """
        video_record = record_fname is not None
        recorder = None if not video_record else VideoRecorder(self.env, record_fname)

        rewards, cost = [], []
        O, A, reward_sum, done = [self.env.reset()], [], 0, False

        policy.reset()

        # for the whole episode, get action from policy and then act
        for t in range(horizon):
            if video_record:
                recorder.capture_frame()
            # .act() is MPC actually solving limited time optimal control problem
            # for best action given past info and it's planning horizon (plan_hor)
            # pred_next_state is num_particles x observation space
            a, c, pred_trajs = policy.act(O[t], t, get_pred_cost=True) #O[t] is current state

            A.append(a)

            if self.noise_stddev is None:
                # jsw Using environment to actually step the obs, reward, and info
                obs, reward, done, info = self.env.step(A[t])
            else:
                # jsw: otherwise, you action has some noise on it, and you step using actually env
                action = A[t] + np.random.normal(loc=0, scale=self.noise_stddev, size=[self.dU])
                action = np.minimum(np.maximum(action, self.env.action_space.low), self.env.action_space.high)
                obs, reward, done, info = self.env.step(action)
            O.append(obs)


            #Print the observed vs. predicted next state. time=0 is last true obs state in MPC
            # if pred_trajs is not None:
            #     pred_next_state = pred_trajs[1, :, :, :].squeeze()
            #     # has 20 particles, 33 observations. Take the mean
            #     print(f"Predicted next_goal_dist: {pred_next_state[:, 0]}, actual: {obs[0]}")

            reward_sum += reward
            rewards.append(reward)
            if 'cost' in info:
                cost.append(info['cost'])
            if done:
                if t < 999:
                    print(f"finished early at t{t}")
                break

        if video_record:
            recorder.capture_frame()
            recorder.close()

        return {
            "obs": np.array(O),
            "ac": np.array(A),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
            "cost": cost
        }
