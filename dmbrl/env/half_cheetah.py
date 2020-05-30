from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


# Jsw to make manually:
# cheetah = gym.make("HalfCheetah-v1")
# >>> cheetah.action_space
# Box(6,)
# >>> cheetah.observation_space (seemt to be skipping one in gym)
# Box(17,)
# They literally just copied over the half_cheetah.py
# https://github.com/openai/gym/blob/master/gym/envs/mujoco/half_cheetah.py
# Description of those dim from xml: https://github.com/openai/gym/blob/master/gym/envs/mujoco/assets/half_cheetah.xml
# State-Space (name/joint/parameter):
#     - rootx     slider      position (m)
#     - rootz     slider      position (m)
#     - rooty     hinge       angle (rad)
#     - bthigh    hinge       angle (rad)
#     - bshin     hinge       angle (rad)
#     - bfoot     hinge       angle (rad)
#     - fthigh    hinge       angle (rad)
#     - fshin     hinge       angle (rad)
#     - ffoot     hinge       angle (rad)
#     - rootx     slider      velocity (m/s)
#     - rootz     slider      velocity (m/s)
#     - rooty     hinge       angular velocity (rad/s)
#     - bthigh    hinge       angular velocity (rad/s)
#     - bshin     hinge       angular velocity (rad/s)
#     - bfoot     hinge       angular velocity (rad/s)
#     - fthigh    hinge       angular velocity (rad/s)
#     - fshin     hinge       angular velocity (rad/s)
#     - ffoot     hinge       angular velocity (rad/s)
# Actuators (name/actuator/parameter):
#     - bthigh    hinge       torque (N m)
#     - bshin     hinge       torque (N m)
#     - bfoot     hinge       torque (N m)
#     - fthigh    hinge       torque (N m)
#     - fshin     hinge       torque (N m)
#     - ffoot     hinge       torque (N m)
class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def _step(self, action):
        self.prev_qpos = np.copy(self.model.data.qpos.flat)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        # Regularizing term on the
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = ob[0] - 0.0 * np.square(ob[2])
        reward = reward_run + reward_ctrl

        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            (self.model.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
            self.model.data.qpos.flat[1:],
            self.model.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(loc=0, scale=0.001, size=self.model.nq)
        qvel = self.init_qvel + np.random.normal(loc=0, scale=0.001, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.model.data.qpos.flat)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55
