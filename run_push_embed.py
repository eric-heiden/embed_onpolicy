#!/usr/bin/env python3

from datetime import datetime
from operator import itemgetter
from typing import Callable

import gym
import sys
import numpy as np
import os.path as osp
import tensorflow as tf
import imageio
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw

from curriculum import ReverseCurriculum, BasicCurriculum

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))
sys.path.insert(0, osp.join(osp.dirname(__file__), 'garage'))

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

from garage.envs.mujoco.sawyer import PushEnv
from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnvWrapper, SawyerEnv

from policies import MlpEmbedPolicy
import ppo2embed

SEED = 1234
TRAJ_SIZE = 120
TASKS = ["up"]  # , "down", "left", "right"]
CONTROL_MODE = "position_control"
EASY_GRIPPER_INIT = True
RANDOMIZE_START_POS = True

# use Beta distribution for policy, Gaussian otherwise
USE_BETA = False
ACTION_SCALE = 5. if USE_BETA else 1.
SKIP_STEPS = 1
USE_EMBEDDING = False


def unwrap_env(env: VecNormalize, id: int = 0):
    return env.unwrapped.envs[id].env


def train(num_timesteps, seed, log_folder):
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    task_space = gym.spaces.Box(low=0, high=1, shape=(len(TASKS),), dtype=np.float32)
    latent_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    env_fn = lambda task: VecNormalize(
        DummyVecEnv([lambda: SawyerEnvWrapper(PushEnv(direction=TASKS[task],
                                                      control_method=CONTROL_MODE,
                                                      easy_gripper_init=EASY_GRIPPER_INIT,
                                                      randomize_start_pos=RANDOMIZE_START_POS,
                                                      action_scale=ACTION_SCALE,
                                                      max_episode_steps=TRAJ_SIZE,
                                                      skip_steps=SKIP_STEPS))]),
        ob=False, ret=False
    )

    def plot_traj(fig, where, task, batch_tuple_size, batches, colormap):
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d
        ax = fig.add_subplot(where, projection='3d')
        ax.set_title("Task %i (%s)" % (task + 1, TASKS[task]))
        ax.grid()
        ax.set_aspect("equal")
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        LIMITS = [0, .5, -.2]

        # for i, batch in enumerate(batches):
        #     obs, tasks, returns, masks, actions, values, neglogpacs, latents, epinfos, \
        #     inference_means, inference_stds = tuple(batch)
        #     ax.plot(obs[:, 1], obs[:, 2], color=colormap(i * 1. / len(batches)),
        #             zorder=2, linewidth=.5, zs=LIMITS[0], zdir='x')
        #     ax.plot(obs[:, 0], obs[:, 2], color=colormap(i * 1. / len(batches)),
        #             zorder=2, linewidth=.5, zs=LIMITS[1], zdir='y')
        #     ax.plot(obs[:, 0], obs[:, 1], color=colormap(i * 1. / len(batches)),
        #             zorder=2, linewidth=.5, zs=LIMITS[2], zdir='z')

        ax.set_xlim([0, 1])
        ax.set_ylim([-.5, .5])
        ax.set_zlim([-.2, .6])

    def render_robot(task: int, iteration: int):
        font = ImageFont.truetype("Consolas.ttf", 32)
        orange = np.array([255, 163, 0])
        red = np.array([255, 0, 0])
        blue = np.array([20, 163, 255])
        width_factor = 1. / TRAJ_SIZE * 512
        lower_part = 512 // 5
        max_reward = 1.

        def render(env: SawyerEnv, obs, actions, values, rewards, infos):
            env.render_camera = "camera_side"
            img_left = env.render(mode='rgb_array')
            env.render_camera = "camera_topdown"

            img_center = env.render(mode='rgb_array')
            # render rewards
            img_center[-lower_part, :10] = orange
            img_center[-lower_part, -10:] = orange
            if TRAJ_SIZE < 512:
                p_rew_x = 0
                for j, r in enumerate(rewards):
                    rew_x = int(j * width_factor)
                    if r < 0:
                        rew_y = int(-r / max_reward * lower_part) if -r <= max_reward else lower_part
                        color = blue if infos[j]["episode"]["grasped"] else red
                    else:
                        rew_y = int(r / max_reward * lower_part) if r <= max_reward else lower_part
                        color = blue if infos[j]["episode"]["grasped"] else orange
                    img_center[-rew_y - 1:, p_rew_x:rew_x] = color
                    img_center[-rew_y - 1:, p_rew_x:rew_x] = color
                    p_rew_x = rew_x
            else:
                for j, r in enumerate(rewards):
                    rew_x = int(j * width_factor)
                    if r < 0:
                        rew_y = int(-r / max_reward * lower_part) if -r <= max_reward else lower_part
                        color = blue if infos[j]["episode"]["grasped"] else red
                    else:
                        rew_y = int(r / max_reward * lower_part) if r <= max_reward else lower_part
                        color = blue if infos[j]["episode"]["grasped"] else orange
                    img_center[-rew_y - 1:, rew_x] = color
                    img_center[-rew_y - 1:, rew_x] = color

            env.render_camera = "camera_front"
            img_right = env.render(mode='rgb_array')
            img_left = Image.fromarray(np.uint8(img_left))
            draw_left = ImageDraw.Draw(img_left)
            draw_left.text((20, 20), "Task %i (%s)" % (task + 1, TASKS[task]), fill="black", font=font)
            img_right = Image.fromarray(np.uint8(img_right))
            draw_right = ImageDraw.Draw(img_right)
            draw_right.text((20, 20), "Iteration %i" % iteration, fill="black", font=font)
            return np.hstack((np.array(img_left), np.array(img_center), np.array(img_right)))

        return render

    set_global_seeds(seed)
    policy = lambda *args, **kwargs: MlpEmbedPolicy(*args, **kwargs, use_beta=USE_BETA)
    ppo2embed.learn(policy=policy,
                    env_fn=env_fn,
                    unwrap_env=unwrap_env,
                    task_space=task_space,
                    latent_space=latent_space,
                    traj_size=TRAJ_SIZE,
                    nbatches=10,
                    lam=0.98,
                    gamma=0.995,
                    policy_entropy=10.,  # .01,  # 0.1,
                    embedding_entropy=-1e3,  # -0.01,  # 0.01,
                    inference_coef=0, #.001,  # 0.03,  # .001,
                    inference_opt_epochs=3,  # 3,
                    inference_horizon=3,
                    log_interval=1,
                    em_hidden_layers=(2,),
                    pi_hidden_layers=(16, 16),
                    vf_hidden_layers=(16, 16),
                    vf_coef=1,
                    inference_hidden_layers=(2,),
                    render_fn=render_robot,
                    lr=5e-3,
                    cliprange=0.2,
                    seed=seed,
                    total_timesteps=num_timesteps,
                    plot_folder=osp.join(log_folder, "plots"),
                    plot_interval=15,
                    render_interval=15,
                    render_fps=60,
                    use_embedding=USE_EMBEDDING,
                    traj_plot_fn=plot_traj,
                    log_folder=log_folder,
                    curriculum_fn=BasicCurriculum)


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_folder = osp.join(osp.dirname(__file__), 'log/push_pos_embed_%i_%s' % (SEED, timestamp))
    print("Logging to %s." % log_folder)
    logger.configure(dir=log_folder, format_strs=['stdout', 'log', 'csv'])
    train(num_timesteps=1e6, seed=SEED, log_folder=log_folder)


if __name__ == '__main__':
    main()
