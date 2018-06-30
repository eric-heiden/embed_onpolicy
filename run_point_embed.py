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


SEED = 12345

# use Beta distribution for policy, Gaussian otherwise
USE_BETA = True


def train(num_timesteps, seed, log_folder):
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    task_space = gym.spaces.Box(low=0, high=1, shape=(len(TASKS),))
    latent_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))

    def make_env():
        env = PointEnv()
        return env

    env = DummyVecEnv([make_env])
    # env = VecNormalize(env, ob=True, ret=False, cliprew=200)

    set_global_seeds(seed)
    policy = lambda *args, **kwargs: MlpEmbedPolicy(*args, **kwargs, use_beta=USE_BETA)
    model = ppo2embed.learn(policy=policy,
                            env=env,
                            task_space=task_space,
                            latent_space=latent_space,
                            # nsteps=1000,
                            # nminibatches=5,
                            traj_size=30,
                            nbatches=3,
                            lam=0.95,
                            gamma=0.99,
                            pi_opt_epochs=30,
                            inference_opt_epochs=10,
                            log_interval=1,
                            policy_entropy=0.01,
                            embedding_entropy=0.,
                            inference_coef=1.,
                            inference_horizon=3,
                            lr=5e-3,
                            cliprange=0.2,
                            seed=seed,
                            total_timesteps=num_timesteps,
                            plot_folder=osp.join(log_folder, "plots"),
                            log_folder=log_folder)

    return model, env


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_folder = osp.join(osp.dirname(__file__), 'log/point_embed_%i_%s' % (SEED, timestamp))
    print("Logging to %s." % log_folder)
    logger.configure(dir=log_folder,
                     format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    model, env = train(num_timesteps=3e6, seed=SEED, log_folder=log_folder)

    logger.log("Running trained model")
    for _ in range(20):
        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        latent = [model.get_latent(e.task) for e in env.envs]
        for _ in range(50):
            actions = model.step(latent, obs)[0]
            obs[:], _, done, _ = env.step(actions)
            env.render()
            if done:
                print("Done")
                break


if __name__ == '__main__':
    main()
