#!/usr/bin/env python3

from datetime import datetime
import gym
import sys
import numpy as np
import os.path as osp
import tensorflow as tf

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from point3d_env import Point3dEnv, TASKS
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
    latent_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))

    env = DummyVecEnv([lambda: Point3dEnv()])
    # env = VecNormalize(env, ob=True, ret=False, cliprew=200)

    def plot_traj(fig, where, task, batch_tuple_size, batches, colormap):
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d
        ax = fig.add_subplot(where, projection='3d')
        ax.set_title("Task %i" % (task + 1))
        ax.grid()
        ax.set_aspect("equal")
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # ax.scatter([0], [0], [0], s=16, c='black')
        # ax.scatter([TASKS[task][0]], [TASKS[task][1]], [TASKS[task][2]], s=16, c='orange')

        ax.scatter([0], [0], s=9, c='black', zs=-5, zdir='x', zorder=1)
        ax.scatter([TASKS[task][1]], [TASKS[task][2]], s=16, c='orange', zs=-5, zdir='x', zorder=1)

        ax.scatter([0], [0], s=9, c='black', zs=5, zdir='y', zorder=1)
        ax.scatter([TASKS[task][0]], [TASKS[task][2]], s=16, c='orange', zs=5, zdir='y', zorder=1)

        ax.scatter([0], [0], s=9, c='black', zs=-5, zdir='z', zorder=1)
        ax.scatter([TASKS[task][0]], [TASKS[task][1]], s=16, c='orange', zs=-5, zdir='z', zorder=1)

        for i, batch in enumerate(batches):
            bs = tuple([np.array([batch[i][k] for i in range(len(batch))]) for k in range(batch_tuple_size)])
            obs, tasks, returns, masks, actions, values, neglogpacs, latents, epinfos, \
                inference_means, inference_stds = bs
            # ax.plot([0] + obs[:, 0], [0] + obs[:, 1], [0] + obs[:, 2], color=colormap(i * 1. / len(batches)),
            #              zorder=2, linewidth=.5, marker='o', markersize=0.5, alpha=0.1)
            ax.plot([0] + obs[:, 1], [0] + obs[:, 2], color=colormap(i * 1. / len(batches)),
                         zorder=2, linewidth=.5, zs=-5, zdir='x')
            ax.plot([0] + obs[:, 0], [0] + obs[:, 2], color=colormap(i * 1. / len(batches)),
                         zorder=2, linewidth=.5, zs=5, zdir='y')
            ax.plot([0] + obs[:, 0], [0] + obs[:, 1], color=colormap(i * 1. / len(batches)),
                         zorder=2, linewidth=.5, zs=-5, zdir='z')

        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_zlim([-4, 4])

    set_global_seeds(seed)
    policy = lambda *args, **kwargs: MlpEmbedPolicy(*args, **kwargs, use_beta=USE_BETA)
    model = ppo2embed.learn(policy=policy,
                            env=env,
                            task_space=task_space,
                            latent_space=latent_space,
                            traj_size=40,
                            nbatches=4,
                            lam=0.95,
                            gamma=0.99,
                            inference_opt_epochs=5,
                            log_interval=1,
                            policy_entropy=0.1,
                            embedding_entropy=0.01,
                            inference_coef=.001,
                            inference_horizon=3,
                            em_hidden_layers=(16,),
                            pi_hidden_layers=(32, 32),
                            vf_hidden_layers=(32, 32),
                            inference_hidden_layers=(16,),
                            lr=5e-3,
                            cliprange=0.2,
                            seed=seed,
                            total_timesteps=num_timesteps,
                            plot_folder=osp.join(log_folder, "plots"),
                            traj_plot_fn=plot_traj,
                            log_folder=log_folder)

    return model, env


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_folder = osp.join(osp.dirname(__file__), 'log/point_embed_%i_%s' % (SEED, timestamp))
    print("Logging to %s." % log_folder)
    logger.configure(dir=log_folder, format_strs=['stdout', 'log', 'csv'])
    train(num_timesteps=5e7, seed=SEED, log_folder=log_folder)


if __name__ == '__main__':
    main()
