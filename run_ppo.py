import tensorflow as tf
import gym
import numpy as np
from baselines.ppo2 import ppo2
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2.policies import MlpPolicy
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from sawyer_env import SawyerEnvWrapper
from sawyer_down_env import DownEnv
# from gym.envs.robotics import FetchPickAndPlaceEnv


class GoalEnvWrapper:

    def __init__(self, env):
        self.env = env
        self.len = 0
        self.rew = 0

    @property
    def observation_space(self):
        obs = self.env._get_obs()
        return gym.spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32')

    def reset(self):
        self.len = 0
        self.rew = 0
        return self.env.reset()['observation']

    def step(self, action):
        next_obs, r, done, info = self.env.step(action)
        next_obs = next_obs['observation']
        self.rew += r
        self.len += 1
        if self.len >= 1000:
            info = dict(episode=dict(r=self.rew, l=self.len))
            print('reset')
            next_obs = self.reset()
            r = 0
        return next_obs, r, False, info

    @property
    def action_space(self):
        return self.env.action_space

    def render(self, mode='human'):
        self.env.render(mode)


def ppo():
    def make_env():
        env = SawyerEnvWrapper(DownEnv(for_her=False))
        return env

    tf.Session().__enter__()
    env = VecNormalize(DummyVecEnv([make_env]))
    policy = MlpPolicy
    model = ppo2.learn(policy=policy, env=env, nsteps=4000, nminibatches=1,
                       lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
                       ent_coef=0.0, lr=3e-4, cliprange=0.2, total_timesteps=1e8)

    return model

ppo()