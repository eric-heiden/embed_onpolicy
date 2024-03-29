#!/usr/bin/env python3

import sys
import os.path as osp
from datetime import datetime

import numpy as np
import tensorflow as tf

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))

from baselines import bench, logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from point_env import PointEnv
from policies import MlpPolicy
import ppo2


SEED = 123


def train(num_timesteps, seed):
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    def make_env():
        env = PointEnv()
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, ret=False, cliprew=200)

    set_global_seeds(seed)
    policy = MlpPolicy
    model = ppo2.learn(policy=policy, env=env, nsteps=100, nminibatches=25,
                       lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
                       ent_coef=0.0,
                       lr=3e-4,
                       cliprange=0.2,
                       total_timesteps=num_timesteps)

    return model, env


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_folder = osp.join(osp.dirname(__file__), 'log/point_%i_%s' % (SEED, timestamp))
    print("Logging to %s." % log_folder)
    logger.configure(dir=log_folder,
                     format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    model, env = train(num_timesteps=1e4, seed=SEED)

    logger.log("Running trained model")
    for _ in range(20):
        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        for _ in range(50):
            actions = model.step(obs)[0]
            obs[:], _, done, _ = env.step(actions)
            env.render()
            if done:
                print("Done")
                break


if __name__ == '__main__':
    main()
