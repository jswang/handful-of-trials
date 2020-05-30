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


class SafetyPointGoal1ConfigModule:
    ENV_NAME = 'Safexp-PointGoal1-v0'
    # copied from pusher
    TASK_HORIZON = 150
    NTRAIN_ITERS = 100
    NROLLOUTS_PER_ITER = 1
    # from pusher
    PLAN_HOR = 25
    # 1 state points, 2 action points
    MODEL_IN, MODEL_OUT = 3, 2
    GP_NINDUCING_POINTS = 200

    def __init__(self):

        config = {
            'placements_extents': [-1.5,-1.5, 1.5, 1.5],
            'robot_base': 'xmls/point.xml',
            'observe_goal_dist': True,
            # 'observe_goal_lidar': True,
            # 'observe_box_lidar': True,
            # 'observe_hazards': True,
            # 'observe_vases': True,
            'observe_goal_lidar': False,
            'observe_box_lidar': False,
            'observe_hazards': False,
            'observe_vases': False,

            'vision_size': (60, 40),
            'vision_render_size': (300, 200),
            'lidar_num_bins': 16,
            'lidar_max_dist': 3,
            'goal_keepout': 0.305,
            'reward_circle': 0.1,
            'sensors_obs':[], # ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
            'constrain_hazards': True,
            'hazards_num': 8,
            'hazards_keepout': 0.18,
            'hazards_size': 0.2,
            'vases_num': 1,
            'vases_sink': 4e-05,
            'vases_displace_threshold': 0.001,
            'vases_velocity_threshold': 0.0001,
            '_seed': None
        }
        self.ENV = Engine(config)

        #settings from pusher
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.compat.v1.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2500
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    # Safety gym already transforms angles into cosines and sines

    # These make the NN learn deltas on next state rather than true next state
    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred
    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    # Just using self to get constants, can't store any state b/c this function
    # gets parallelized
    def obs_cost_fn(next_obs, cur_obs):
        # Components of observation, from safety-gym engine.py obs()
        # ['goal_dist', 'goal_lidar', 'hazards_lidar', 'vases_lidar']
        if isinstance(next_obs, np.ndarray):
            goal_dist = -np.log(next_obs[0, :]) # np.exp(-self.dist_goal())
            prev_goal_dist = -np.log(cur_obs[0, :]) # np.exp(-self.dist_goal())
        else:
            goal_dist = -tf.log(next_obs[0, :]) # np.exp(-self.dist_goal())
            prev_goal_dist = -tf.log(cur_obs[0, :]) # np.exp(-self.dist_goal())

        reward = 0
        # dense reward for moving closer to goal
        reward += (prev_goal_dist - goal_dist) # * self.ENV.config['reward_distance']
        # reward for hitting the goal
        if goal_dist <= self.goal_size:
            reward += 1.0 #self.reward_goal

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
        if not model_init_cfg.get("load_model", False):
            model.add(FC(500, input_dim=self.MODEL_IN, activation="swish", weight_decay=0.000025))
            model.add(FC(500, activation="swish", weight_decay=0.00005))
            model.add(FC(500, activation="swish", weight_decay=0.000075))
            model.add(FC(500, activation="swish", weight_decay=0.000075))
            model.add(FC(self.MODEL_OUT, weight_decay=0.0001))
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