import sys
import os.path as osp
import tensorflow as tf

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))
sys.path.insert(0, osp.join(osp.dirname(__file__), 'garage'))

from baselines.ppo2 import ppo2
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2.policies import MlpPolicy
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from sawyer_env import SawyerEnvWrapper
from sawyer_down_env import DownEnv


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
