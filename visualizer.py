import numpy as np

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

import point_env
from ppo2embed import Model

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab
import os.path as osp


class Visualizer(object):
    def __init__(self, model: Model, env: DummyVecEnv, plot_folder: str):
        self.model = model
        self.env = env
        self.plot_folder = plot_folder

    def visualize(self, update: int, batches):
        nenv = self.env.num_envs
        ntasks = len(point_env.TASKS)

        fig = plt.figure()
        fig.suptitle(
            ('Iteration %i' % update) + (' (using %s distribution)' % ('Normal', 'Beta')[int(self.model.use_beta)]))

        gs0 = gridspec.GridSpec(1, ntasks)

        for task in range(ntasks):
            gs00 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[task])


            obs, tasks, returns, masks, actions, values, neglogpacs, states, latents, inference_log_likelihoods = batches[task]

            ax = plt.Subplot(fig, gs00[0])
            ax.set_title(str(tasks[0]))  # "Task % i" % (t + 1))
            ax.grid()
            ax.set_xlim([-4, 4])
            ax.set_ylim([-4, 4])
            ax.set_aspect('equal')

            goal = plt.Circle(point_env.TASKS[task], radius=point_env.MIN_DIST, color='orange')
            ax.add_patch(goal)

            embedding_mean, embedding_std = tuple(self.model.act_model.embedding_params([tasks[0]]))
            # inference_mean, inference_std = tuple(self.model.inference_model.embedding_params())

            print(embedding_mean, embedding_std)

        #     ax1 = plt.Subplot(f, gs00[:-1, :])
        #     f.add_subplot(ax1)
        #     ax2 = plt.Subplot(f, gs00[-1, :-1])
        #     f.add_subplot(ax2)
        #     ax3 = plt.Subplot(f, gs00[-1, -1])
        #     f.add_subplot(ax3)
        #
        # gs01 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[1])
        #
        # ax4 = plt.Subplot(f, gs01[:, :-1])
        # f.add_subplot(ax4)
        # ax5 = plt.Subplot(f, gs01[:-1, -1])
        # f.add_subplot(ax5)
        # ax6 = plt.Subplot(f, gs01[-1, -1])
        # f.add_subplot(ax6)

        nsteps = 40
        nsamples = 10

        colormap = lambda x: matplotlib.cm.get_cmap("winter")(1. - x)

        # fig, axes = plt.subplots(figsize=(ntasks * 4, 8), nrows=2, ncols=ntasks, sharey='row')
        # if ntasks == 1:
        #     axes = [[axes[0]], [axes[1]]]
        # for t in range(ntasks):
        #     one_hot = np.zeros((self.model.task_space.shape[0],))
        #     one_hot[t] = 1
        #     onehots = [one_hot]
        #
        #     ax = axes[0][t]
        #
        #     # ax.scatter([point_env.TASKS[t][0]], [point_env.TASKS[t][1]], s=50, c='r')
        #
        #     ax_latent = axes[1][t]
        #     ax_latent.grid()
        #     ticks = np.arange(0, self.model.latent_space.shape[0], 1)
        #     ax_latent.set_xticks(ticks)
        #
        #     for sample in range(nsamples):
        #         obs = np.zeros((nenv,) + self.env.observation_space.shape, dtype=self.env.observation_space.dtype.name)
        #         obs[:] = self.env.reset()
        #         for pointEnv in self.env.envs:
        #             pointEnv._task = t  # TODO expose via setter
        #             pointEnv._goal = point_env.TASKS[t]
        #         dones = [False for _ in range(nenv)]
        #         positions = [np.copy(obs[0])]
        #
        #         latents = [self.model.get_latent(t)]
        #         for _ in range(nsteps):
        #             actions, values, mb_states, neglogpacs = self.model.step(latents, obs, self.model.initial_state, dones)
        #             # actions, values, mb_states, neglogpacs = self.model.step_from_task(
        #             #     onehots, obs, self.model.initial_state, dones)
        #             if not self.model.use_beta:
        #                 actions *= 0.1
        #             # actions = np.clip(actions, self.model.action_space.low[0], self.model.action_space.high[0])
        #             obs[:], rewards, dones, infos = self.env.step(actions)
        #             positions.append(np.copy(obs[0]))
        #             if dones[-1]:
        #                 break
        #         positions = np.array(positions)
        #         positions = np.reshape(positions, (-1, 2))
        #         ax.scatter(positions[:, 0], positions[:, 1], color=colormap(sample * 1. / nsamples), s=2, zorder=2)
        #         ax.plot(positions[:, 0], positions[:, 1], color=colormap(sample * 1. / nsamples), zorder=2)
        #
        #         # visualize latents TODO make actions dependent on these
        #         ax_latent.scatter(ticks, latents, color=colormap(sample * 1. / nsamples))

        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(osp.join(self.plot_folder, 'embed_%05d.png' % update))
        # plt.clf()
