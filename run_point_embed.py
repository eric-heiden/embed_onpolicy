#!/usr/bin/env python3

from datetime import datetime
import gym
import sys
import numpy as np
import os.path as osp
import tensorflow as tf

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))

from baselines import bench, logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from point_env import PointEnv, TASKS
from policies import MlpEmbedPolicy
import ppo2embed


SEED = 123


def train(num_timesteps, seed):
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    task_space = gym.spaces.Box(low=0, high=1, shape=(len(TASKS),))
    latent_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))

    def make_env():
        env = PointEnv()
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True, info_keywords=("episode",))
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, ret=False, cliprew=200)

    set_global_seeds(seed)
    policy = MlpEmbedPolicy
    model = ppo2embed.learn(policy=policy,
                            env=env,
                            task_space=task_space,
                            latent_space=latent_space,
                            nsteps=100,
                            nminibatches=25,
                            lam=0.95,
                            gamma=0.99,
                            noptepochs=10,
                            log_interval=1,
                            ent_coef=0.0,
                            lr=3e-4,
                            cliprange=0.2,
                            total_timesteps=num_timesteps)

    return model, env


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger.configure(dir=osp.join(osp.dirname(__file__), 'log/point_embed_%i_%s' % (SEED, timestamp)),
                     format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    model, env = train(num_timesteps=1e5, seed=SEED)

    logger.log("Running trained model")
    for _ in range(20):
        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        latent = [model.get_latent(e.env.task) for e in env.venv.envs]
        for _ in range(50):
            actions = model.step(latent, obs)[0]
            obs[:], _, done, _ = env.step(actions)
            env.render()
            if done:
                print("Done")
                break


if __name__ == '__main__':
    main()
