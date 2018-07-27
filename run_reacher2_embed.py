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

from sawyer_env import SawyerEnv, SawyerEnvWrapper
from sawyer_reach_env import ReachEnv

from policies import MlpEmbedPolicy
import ppo2embed

SEED = 12345
TRAJ_SIZE = 50
TASKS = [(0.3, -0.3, 0.30), (0.3, 0.3, 0.30)]

# use Beta distribution for policy, Gaussian otherwise
USE_BETA = True


def unwrap_env(env: VecNormalize, id: int = 0):
    return env.unwrapped.envs[id].env


class PickAndPlaceCurriculum(ReverseCurriculum):
    def __init__(self, env_fn: Callable, unwrap_env: Callable, tasks: int, batches: int, **kwargs):
        super().__init__(env_fn, unwrap_env, tasks, batches, **kwargs)

    def initialize(self):
        for task in range(self._tasks):
            for batch in range(self._batches):
                env = self._unwrap_env(self._envs[task][batch])  # type: DownEnv
                # env.set_start_position(np.array(env.goal[:3]) + np.array([0, 0, .2]))
                env.set_start_position(np.concatenate([np.array(env.goal[:2]), [0.05]]))

    def resample(self, task, batch):
        env = self._unwrap_env(self._envs[task][batch])
        start_pos = self._starts[task].copy()
        curr_pos = env.start_position
        if curr_pos[2] < env.goal[2]:
            pos = np.array(curr_pos) + np.array([0, 0, 0.005])
            distance = np.inf
        else:
            print("Task %i at batch id %i was fixed to the right z position. (Update %i/%i)" %
                  (task, batch, self._updates[task][batch] + 1, self._max_progress))
            self._updates[task][batch] = self._max_progress
            env.set_start_position(start_pos)
            return
        # else:
        #     samples = []
        #     for _ in range(self._action_samples):
        #         env.set_position(curr_pos)
        #         for _ in range(self._delta_steps):
        #             env.step(env.action_space.sample())
        #         samples.append((np.linalg.norm(start_pos - env.position), env.position))
        #     samples = sorted(samples, key=itemgetter(0))
        #     distance, pos = samples[0]
        if distance < self._distance_threshold or self._updates[task][batch] == self._max_progress - 1:
            print("Task %i at batch id %i has reached the start position. (Update %i/%i)" %
                  (task, batch, self._updates[task][batch] + 1, self._max_progress))
            env.set_start_position(start_pos)
            self._updates[task][batch] = self._max_progress
        else:
            self._updates[task][batch] += 1
            if self._updates[task][batch] % 5 == 0:
                print("Updating curriculum for task %i at batch id %i. (Update %i/%i)" %
                      (task, batch, self._updates[task][batch] + 1, self._max_progress))
                env.set_start_position(pos)


def train(num_timesteps, seed, log_folder):
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    task_space = gym.spaces.Box(low=0, high=1, shape=(len(TASKS),), dtype=np.float32)
    latent_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    env_fn = lambda task: VecNormalize(
        DummyVecEnv([lambda: SawyerEnvWrapper(ReachEnv(goal_position=TASKS[task]))]),
        ob=False, ret=False
    )
    env_ = unwrap_env(env_fn(task=0))
    # print("Start position:", env_.start_position)

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

        # start_pos = env_.start_position

        LIMITS = [0, .5, 0]

        # ax.scatter([start_pos[1]], [start_pos[2]], s=9, c='black', zs=LIMITS[0], zdir='x', zorder=1)
        # ax.scatter([TASKS[task][1]], [TASKS[task][2]], s=16, c='orange', zs=LIMITS[0], zdir='x', zorder=1)

        # ax.scatter([start_pos[0]], [start_pos[2]], s=9, c='black', zs=LIMITS[1], zdir='y', zorder=1)
        # ax.scatter([TASKS[task][0]], [TASKS[task][2]], s=16, c='orange', zs=LIMITS[1], zdir='y', zorder=1)

        # ax.scatter([start_pos[0]], [start_pos[1]], s=9, c='black', zs=LIMITS[2], zdir='z', zorder=1)
        # ax.scatter([TASKS[task][0]], [TASKS[task][1]], s=16, c='orange', zs=LIMITS[2], zdir='z', zorder=1)

        for i, batch in enumerate(batches):
            # bs = tuple([np.array([batch[i][k] for i in range(len(batch))]) for k in range(batch_tuple_size)])
            obs, tasks, returns, masks, actions, values, neglogpacs, latents, epinfos, \
            inference_means, inference_stds = tuple(batch)
            # ax.plot([0] + obs[:, 0], [0] + obs[:, 1], [0] + obs[:, 2], color=colormap(i * 1. / len(batches)),
            #              zorder=2, linewidth=.5, marker='o', markersize=0.5, alpha=0.1)
            ax.plot(obs[:, 1], obs[:, 2], color=colormap(i * 1. / len(batches)),
                    zorder=2, linewidth=.5, zs=LIMITS[0], zdir='x')
            ax.plot(obs[:, 0], obs[:, 2], color=colormap(i * 1. / len(batches)),
                    zorder=2, linewidth=.5, zs=LIMITS[1], zdir='y')
            ax.plot(obs[:, 0], obs[:, 1], color=colormap(i * 1. / len(batches)),
                    zorder=2, linewidth=.5, zs=LIMITS[2], zdir='z')

        ax.set_xlim([0, 1])
        ax.set_ylim([-.5, .5])
        ax.set_zlim([0, 1])

    def render_robot(task: int, iteration: int):
        font = ImageFont.truetype("Consolas.ttf", 32)
        orange = np.array([255, 163, 0])
        red = np.array([255, 0, 0])
        blue = np.array([20, 163, 255])
        width_factor = 1. / TRAJ_SIZE * 512
        lower_part = 512 // 5
        max_reward = 1.

        def render(env: SawyerEnv, obs, actions, values, rewards, infos):
            # assert(env.envs[0]._task == task)
            # print("TASK %i == %i   %s" % (env.envs[0]._task, task, str(env.envs[0]._goal)))
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
                        color = blue if infos[j]["episode"]["grasped"] else red
                        img_center[-1:, p_rew_x:rew_x] = color
                        img_center[-1:, p_rew_x:rew_x] = color
                    else:
                        rew_y = int(r / max_reward * lower_part)
                        color = blue if infos[j]["episode"]["grasped"] else orange
                        img_center[-rew_y - 1:, p_rew_x:rew_x] = color
                        img_center[-rew_y - 1:, p_rew_x:rew_x] = color
                    p_rew_x = rew_x
            else:
                for j, r in enumerate(rewards):
                    rew_x = int(j * width_factor)
                    if r < 0:
                        color = blue if infos[j]["episode"]["grasped"] else red
                        img_center[-1:, rew_x] = color
                        img_center[-1:, rew_x] = color
                    else:
                        rew_y = int(r / max_reward * lower_part)
                        color = blue if infos[j]["episode"]["grasped"] else orange
                        img_center[-rew_y - 1:, rew_x] = color
                        img_center[-rew_y - 1:, rew_x] = color

            env.render_camera = "camera_front"
            img_right = env.render(mode='rgb_array')
            img_left = Image.fromarray(np.uint8(img_left))
            draw_left = ImageDraw.Draw(img_left)
            draw_left.text((20, 20), "Task %i" % (task + 1), fill="black", font=font)
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
                    nbatches=50,
                    lam=0.95,
                    gamma=0.99,
                    policy_entropy=0.1, #.01,  # 0.1,
                    embedding_entropy=-0.01,  # -0.01,  # 0.01,
                    inference_coef=0.01,  # .001,
                    inference_opt_epochs=3,  # 3,
                    inference_horizon=3,
                    log_interval=1,
                    em_hidden_layers=(2,),
                    pi_hidden_layers=(32, 32),
                    vf_hidden_layers=(32, 32),
                    inference_hidden_layers=(16,),
                    render_fn=render_robot,
                    lr=5e-3,
                    cliprange=0.2,
                    seed=seed,
                    total_timesteps=num_timesteps,
                    plot_folder=osp.join(log_folder, "plots"),
                    plot_interval=20,
                    render_interval=20,
                    traj_plot_fn=plot_traj,
                    log_folder=log_folder,
                    curriculum_fn=BasicCurriculum)
                    # curriculum_fn=lambda *args, **kwargs:
                    # PickAndPlaceCurriculum(*args, delta_steps=1, return_threshold=100., max_progress=100, **kwargs))


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_folder = osp.join(osp.dirname(__file__), 'log/reacher2_embed_%i_%s' % (SEED, timestamp))
    print("Logging to %s." % log_folder)
    logger.configure(dir=log_folder, format_strs=['stdout', 'log', 'csv'])
    train(num_timesteps=1e6, seed=SEED, log_folder=log_folder)

    # env = TaskPickAndPlaceEnv(control_method="position_control")
    # video = []
    # font = ImageFont.truetype("Consolas.ttf", 32)
    # for t in tqdm(range(len(TASKS))):
    #     env.select_task(t)
    #     pos = env.reset()
    #     print(TASKS[t])
    #     # env.set_position(TASKS[t])
    #     env.render()
    #     # action = (0., 0., 0.1)
    #     for i in range(15):
    #         # env.step(np.zeros(3))
    #         env.step(env.action_space.sample())
    #         env.render_camera = "camera_side"
    #         # env.render()
    #         img_left = env.render(mode='rgb_array')
    #         env.render_camera = "camera_topdown"
    #         img_center = env.render(mode='rgb_array')
    #         env.render_camera = "camera_front"
    #         img_right = env.render(mode='rgb_array')
    #         img_left = Image.fromarray(np.uint8(img_left))
    #         draw_left = ImageDraw.Draw(img_left)
    #         draw_left.text((20, 20), "Task %i" % t, fill="black", font=font)
    #         img_right = Image.fromarray(np.uint8(img_right))
    #         draw_right = ImageDraw.Draw(img_right)
    #         draw_right.text((20, 20), "Iteration %i" % i, fill="black", font=font)
    #         video.append(np.hstack((np.array(img_left), np.array(img_center), np.array(img_right))))
    # print("Saving video...")
    # imageio.mimsave("test.mp4", video, fps=30)


if __name__ == '__main__':
    main()
