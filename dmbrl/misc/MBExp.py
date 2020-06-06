from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from time import time, localtime, strftime

import numpy as np
from scipy.io import savemat
from dotmap import DotMap

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.misc.Agent import Agent


class MBExperiment:
    def __init__(self, params):
        """Initializes class instance.

        Argument:
            params (DotMap): A DotMap containing the following:
                .sim_cfg:
                    .env (gym.env): Environment for this experiment
                    .task_hor (int): Task horizon
                    .stochastic (bool): (optional) If True, agent adds noise to its actions.
                        Must provide noise_std (see below). Defaults to False.
                    .noise_std (float): for stochastic agents, noise of the form N(0, noise_std^2I)
                        will be added.

                .exp_cfg:
                    .ntrain_iters (int): Number of training iterations to be performed.
                    .nrollouts_per_iter (int): (optional) Number of rollouts done between training
                        iterations. Defaults to 1.
                    .ninit_rollouts (int): (optional) Number of initial rollouts. Defaults to 1.
                    .policy (controller): Policy that will be trained.

                .log_cfg:
                    .logdir (str): Parent of directory path where experiment data will be saved.
                        Experiment will be saved in logdir/<date+time of experiment start>
                    .nrecord (int): (optional) Number of rollouts to record for every iteration.
                        Defaults to 0.
                    .neval (int): (optional) Number of rollouts for performance evaluation.
                        Defaults to 1.
        """
        self.env = get_required_argument(params.sim_cfg, "env", "Must provide environment.")
        self.task_hor = get_required_argument(params.sim_cfg, "task_hor", "Must provide task horizon.")
        if params.sim_cfg.get("stochastic", False):
            self.agent = Agent(DotMap(
                env=self.env, noisy_actions=True,
                noise_stddev=get_required_argument(
                    params.sim_cfg,
                    "noise_std",
                    "Must provide noise standard deviation in the case of a stochastic environment."
                )
            ))
        else:
            self.agent = Agent(DotMap(env=self.env, noisy_actions=False))

        self.ntrain_iters = get_required_argument(
            params.exp_cfg, "ntrain_iters", "Must provide number of training iterations."
        )
        self.nrollouts_per_iter = params.exp_cfg.get("nrollouts_per_iter", 1)
        self.ninit_rollouts = params.exp_cfg.get("ninit_rollouts", 1)
        self.policy = get_required_argument(params.exp_cfg, "policy", "Must provide a policy.")

        self.logdir = os.path.join(
            get_required_argument(params.log_cfg, "logdir", "Must provide log parent directory."),
            strftime("%Y-%m-%d--%H:%M:%S", localtime())
        )
        self.nrecord = params.log_cfg.get("nrecord", 0)
        self.neval = params.log_cfg.get("neval", 1)

    def run_experiment(self):
        """Perform experiment.
        """
        os.makedirs(self.logdir, exist_ok=True)

        traj_obs, traj_acs, traj_rets, traj_rews, traj_cost = [], [], [], [], []

        # Perform initial rollouts
        # uses policy.act() to come up with action, should be uniform random
        samples = []
        print("Acting randomly")
        for i in range(self.ninit_rollouts):

            samples.append(
                self.agent.sample(
                    self.task_hor, self.policy
                )
            )
            traj_obs.append(samples[-1]["obs"])
            traj_acs.append(samples[-1]["ac"])
            traj_rews.append(samples[-1]["rewards"])
            traj_cost.append(samples[-1]["cost"])

        # jsw: "Initialize data D with a random controller for one trial"
        if self.ninit_rollouts > 0:
            print("Training on random actions")
            # jsw this trains the NN model for the very first time
            # policy is of type Controller, which MPC inherits from
            self.policy.train(
                [sample["obs"] for sample in samples],
                [sample["ac"] for sample in samples],
                [sample["rewards"] for sample in samples]
            )

        # Training loop:
        # jsw: "for Trial k = 1 to K do:"
        for i in range(self.ntrain_iters):
            print("####################################################################")
            print("Starting training iteration %d." % (i + 1))

            # jsw note that NN model trained above, initialized with action from a random policy
            iter_dir = os.path.join(self.logdir, "train_iter%d" % (i + 1))
            os.makedirs(iter_dir, exist_ok=True)

            samples = []
            for j in range(self.nrecord):
                samples.append(
                    #####################
                    # This call does a lot! uses policy.act() to come up with action
                    # policy.act() solves open loop finite time problem
                    # Uses environment to actually act.
                    #####################
                    self.agent.sample(
                        self.task_hor, self.policy,
                        os.path.join(iter_dir, "rollout%d.mp4" % j)
                    )
                )
            if self.nrecord > 0:
                for item in filter(lambda f: f.endswith(".json"), os.listdir(iter_dir)):
                    os.remove(os.path.join(iter_dir, item))


            # jsw: actually executing action from optimal actions, log it
            # sample() calls Agent.py's sample which gets the best action from the policy, and
            # uses the environment to see what happens when you use that action. it repeats
            # this for the entire horizon. Actually exploring the true environment
            for j in range(max(self.neval, self.nrollouts_per_iter) - self.nrecord):
                samples.append(
                    # jsw for time t = 0 to task horizon
                    self.agent.sample(
                        self.task_hor, self.policy
                    )
                )
            print("Rewards obtained:", [sample["reward_sum"] for sample in samples[:self.neval]])

            traj_obs.extend([sample["obs"] for sample in samples[:self.nrollouts_per_iter]])
            traj_acs.extend([sample["ac"] for sample in samples[:self.nrollouts_per_iter]])
            traj_rets.extend([sample["reward_sum"] for sample in samples[:self.neval]])
            traj_rews.extend([sample["rewards"] for sample in samples[:self.nrollouts_per_iter]])
            traj_cost.extend([sample["cost"] for sample in samples[:self.nrollouts_per_iter]])
            samples = samples[:self.nrollouts_per_iter]

            self.policy.dump_logs(self.logdir, iter_dir)
            savemat(
                os.path.join(self.logdir, "logs.mat"),
                {
                    "observations": traj_obs,
                    "actions": traj_acs,
                    "returns": traj_rets,
                    "rewards": traj_rews,
                    "cost": traj_cost,
                }
            )
            # Delete iteration directory if not used
            if len(os.listdir(iter_dir)) == 0:
                os.rmdir(iter_dir)

            if i < self.ntrain_iters - 1:
                self.policy.train(
                    [sample["obs"] for sample in samples],
                    [sample["ac"] for sample in samples],
                    [sample["rewards"] for sample in samples]
                )
