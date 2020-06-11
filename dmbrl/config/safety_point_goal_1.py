from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from dotmap import DotMap
import gym
import safety_gym
from safety_gym.envs.engine import Engine
import copy

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC
import dmbrl.env

# modified version of Safexp-PointGoal1-v0, the agent gets exact
# knowledge of distance to goal, no knowledge of hazards
class SafetyPointGoal1ConfigModule:
    # Number timesteps per episode
    TASK_HORIZON = 1000
    # Number of episodes
    NTRAIN_ITERS = 1000
    # Inital rollouts to train the model with
    NINIT_ROLLOUTS = 20
    # MPC planning horizon
    PLAN_HOR = 25

    NROLLOUTS_PER_ITER = 1
    GP_NINDUCING_POINTS = 200
    # Input: 33 state + 2 actions, output 33 state
    MODEL_IN, MODEL_OUT = 35, 33

    def __init__(self):

        config = {
            'placements_extents': [-1.5,-1.5, 1.5, 1.5],
            'robot_base': 'xmls/point.xml',
            'observe_goal_dist': True,
            'observe_goal_lidar': True,
            # Change these all to true to get cost estimates
            'observe_hazards': True,
            'observe_vases': False,

            'vision_size': (60, 40),
            'vision_render_size': (300, 200),
            'lidar_num_bins': 16,
            'lidar_max_dist': 3,
            'goal_keepout': 0.305,
            'sensors_obs': [], #['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
            'constrain_hazards': True,
            'hazards_num': 8,
            'hazards_keepout': 0.18,
            'hazards_size': 0.2,
            '_seed': 0 # TODO: consider taking out
        }
        self.ENV = Engine(config)

        #settings from cartpole
        cfg = tf.compat.v1.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.compat.v1.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 5, "initial_epochs": 35}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2000
            },
            "CEM": {
                "popsize": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1
            },
            "SafeOpt":{
                "swarmsize":20,
                "max_iters":10,
                "beta":2
            }
        }
    def get_env(self):
        return self.ENV

    # Safety gym already transforms angles into cosines and sines
    # My prediction is just the next state, not a delta on the old state.
    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    # the next state is simply the next observation, no delta being calculated
    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    # This next_obs and obs are tensors with nparticles number of rows
    # next_obs have mean and variance in the the two columns
    # cur_obs just has mean
    @staticmethod
    def obs_cost_fn(next_obs, cur_obs):
        # configs taken from engine.py of safety gym
        CONFIG_REWARD_DISTANCE = 1.0 # reward for reducing distance to the goal
        CONFIG_GOAL_SIZE = 0.3 # radius of the goal
        CONFIG_REWARD_GOAL = 1.0 #reward for reaching the goal
        CONFIG_HAZARDS_COST = 1.0 # Cost for hitting hazard

        # Components of observation, from safety-gym engine.py obs()
        # obs space: ['goal_dist', 'goal_lidar', 'hazards_lidar']
        #               1            16            16
        if isinstance(next_obs, np.ndarray):
            goal_dist = next_obs[:, 0]
            prev_goal_dist = cur_obs[:, 0]
        else:
            goal_dist = next_obs[:, 0]
            prev_goal_dist = cur_obs[:, 0]

        # dense reward for moving closer to goal
        reward = (prev_goal_dist - goal_dist) * CONFIG_REWARD_DISTANCE

        # reward for hitting the goal
        reward += tf.cast(goal_dist <= CONFIG_GOAL_SIZE, dtype=tf.float32) * CONFIG_REWARD_GOAL
        reward = tf.expand_dims(reward, axis=1)

        # Cost incurred from running into a hazard
        if isinstance(next_obs, np.ndarray):
            reward -= np.any(next_obs[:, 17:] == 1) * CONFIG_HAZARDS_COST
        else:
            n, m = next_obs.get_shape()
            reward -= tf.cast(tf.reduce_any(tf.slice(next_obs, [0, 17], [n.value, m.value - 17]) == 1), dtype=tf.float32) * CONFIG_HAZARDS_COST

        # Return cost = -reward
        return -reward

    # In safety point goal, there is no penalization on actions
    def ac_cost_fn(self, acs):
        return 0

    def nn_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS, load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None)
        ))
        # model from cartpole
        if not model_init_cfg.get("load_model", False):
            model.add(FC(500, input_dim=self.MODEL_IN, activation='swish', weight_decay=0.0001))
            model.add(FC(500, activation='swish', weight_decay=0.0005))
            model.add(FC(500, activation='swish', weight_decay=0.0005))
            model.add(FC(self.MODEL_OUT, weight_decay=0.0005))
        model.finalize(tf.compat.v1.train.AdamOptimizer, {"learning_rate": 0.001})
        return model

    def gp_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model",
            kernel_class=get_required_argument(model_init_cfg, "kernel_class", "Must provide kernel class"),
            kernel_args=model_init_cfg.get("kernel_args", {}),
            num_inducing_points=get_required_argument(
                model_init_cfg, "num_inducing_points", "Must provide number of inducing points."
            ),
            sess=self.SESS
        ))
        return model


CONFIG_MODULE = SafetyPointGoal1ConfigModule
