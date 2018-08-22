#!/usr/bin/env python3

import click
import os
import os.path as osp
import sys
from datetime import datetime
from zipfile import ZipFile

import gym
import numpy as np
import tensorflow as tf

from curriculum import BasicCurriculum

from point_env import PointEnv, TASKS, MIN_DIST

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

from policies import MlpEmbedPolicy
import ppo2embed

SEED = 12345
TRAJ_SIZE = 20


# use Beta distribution for policy, Gaussian otherwise
USE_BETA = False
SKIP_STEPS = 4
USE_EMBEDDING = True


def unwrap_env(env: VecNormalize, id: int = 0):
    return env.unwrapped.envs[id]


def train(num_timesteps, seed, log_folder):
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    task_space = gym.spaces.Box(low=0, high=1, shape=(len(TASKS),), dtype=np.float32)
    latent_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    env_fn = lambda task: VecNormalize(
        DummyVecEnv([lambda: PointEnv(task)]),
        ob=False, ret=False
    )

    def plot_traj(fig, where, task, batch_tuple_size, batches, colormap):
        import matplotlib.pyplot as plt
        ax = fig.add_subplot(where)
        ax.set_title("Task %i" % (task + 1))
        ax.grid()
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_aspect('equal')
        goal = plt.Circle(TASKS[task], radius=MIN_DIST, color='orange')
        ax.add_patch(goal)
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)

        for i, batch in enumerate(batches):
            obs, tasks, returns, masks, actions, values, neglogpacs, latents, epinfos, \
            extras = tuple(batch)
            ax.plot([0] + obs[:, 0], [0] + obs[:, 1], color=colormap(i * 1. / len(batches)),
                    zorder=2, linewidth=.5, marker='o', markersize=1)

    set_global_seeds(seed)
    policy = lambda *args, **kwargs: MlpEmbedPolicy(*args, **kwargs, use_beta=USE_BETA)
    ppo2embed.learn(policy=policy,
                    env_fn=env_fn,
                    unwrap_env=unwrap_env,
                    task_space=task_space,
                    latent_space=latent_space,
                    traj_size=TRAJ_SIZE,
                    nbatches=15,
                    lam=0.9,
                    gamma=0.9,
                    policy_entropy=ppo2embed.linear_transition(0.01, 0., 50),  # .01,  # 0.1,
                    embedding_entropy=ppo2embed.linear_transition(-1e3, 1, 150),  # -0.01,  # 0.01,
                    inference_coef=1.,  #.001,  # 0.03,  # .001,
                    inference_opt_epochs=5,  # 3,
                    inference_horizon=6,
                    log_interval=1,
                    em_hidden_layers=(4,),
                    pi_hidden_layers=(32, 32),
                    vf_hidden_layers=(32, 32),
                    inference_hidden_layers=(32, 32),
                    vf_coef=ppo2embed.linear_transition(.1, .2, 400),
                    render_fn=None,
                    lr=ppo2embed.linear_transition(7e-3, 5e-3, 300, continue_beyond_end=True, absolute_min=1e-4),
                    cliprange=0.2,
                    seed=seed,
                    total_timesteps=num_timesteps,
                    plot_folder=osp.join(log_folder, "plots"),
                    plot_interval=15,
                    render_interval=15,
                    save_interval=15,
                    render_fps=60,
                    use_embedding=USE_EMBEDDING,
                    traj_plot_fn=plot_traj,
                    log_folder=log_folder,
                    curriculum_fn=BasicCurriculum)


@click.command()
# @click.argument('config_file', type=str,
#                 default="/home/eric/.deep-rl-docker/embed_onpolicy/log/push_pos_embed_1234_2018-08-18-16-45-32/configuration.pkl")
# @click.option('--checkpoint', type=str, default="latest")
# @click.option('--interactive', type=bool, default=True)
# @click.option('--n_test_rollouts', type=int, default=30)
def main():
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_folder = osp.join(osp.dirname(__file__), 'log/point_embed_%i_%s' % (SEED, timestamp))
    print("Logging to %s." % log_folder)
    plot_folder = osp.join(log_folder, "plots")
    if plot_folder and not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    with ZipFile(osp.join(log_folder, "source.zip"), 'w') as zip:
        for file in [osp.basename(__file__), "ppo2embed.py", "sampler.py", "policies.py", "visualizer.py"]:
            zip.write(file)
    logger.configure(dir=log_folder, format_strs=['stdout', 'log', 'csv'])
    train(num_timesteps=1e6, seed=SEED, log_folder=log_folder)


if __name__ == '__main__':
    main()
