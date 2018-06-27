import copy

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
import os.path as osp


class Visualizer(object):
    def __init__(self, model: Model, env: DummyVecEnv, plot_folder: str):
        self.model = model
        self.env = env
        self.plot_folder = plot_folder

    def visualize(self, update: int, batches):
        ntasks = len(point_env.TASKS)
        nbins = 100
        latent_dim = self.model.latent_space.shape[0]
        colormap = lambda x: matplotlib.cm.get_cmap("winter")(1. - x)

        task_data_prior = [batches[t::ntasks] for t in range(ntasks)]  # batches grouped by task id
        task_data = [[] for _ in range(ntasks)]
        for task in range(ntasks):
            task_samples = []
            for i, batch in enumerate(task_data_prior[task]):
                for t in range(len(batch[0])):
                    sample = [batch[k][t] for k in range(len(batch))]
                    if sample[-1]["l"] == 1 and len(task_samples) > 0:  # epinfo indicates first step
                        task_data[task].append(copy.copy(task_samples))
                        task_samples = [sample]
                    else:
                        task_samples.append(sample)
                if len(task_samples) > 0:
                    task_data[task].append(copy.copy(task_samples))

        fig = plt.figure(figsize=(20, 20))
        fig.suptitle(
            ('Iteration %i' % update) + (' (using %s distribution)' % ('Normal', 'Beta')[int(self.model.use_beta)]))

        gs0 = gridspec.GridSpec(1, ntasks)

        latent_axes = []

        for task in range(ntasks):
            nsamples = float(len(task_data[task]))
            print("Plotting task", task)
            gs00 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[task])
            traj_ax = plt.Subplot(fig, gs00[0])
            traj_ax.set_title("Task % i" % (task + 1))
            traj_ax.grid()
            traj_ax.set_xlim([-5, 5])
            traj_ax.set_ylim([-5, 5])
            traj_ax.set_aspect('equal')
            goal = plt.Circle(point_env.TASKS[task], radius=point_env.MIN_DIST, color='orange')
            traj_ax.add_patch(goal)
            for tl in traj_ax.get_xticklabels() + traj_ax.get_yticklabels():
                tl.set_visible(False)

            for i, batch in enumerate(task_data[task]):
                bs = tuple([np.array([batch[i][k] for i in range(len(batch))]) for k in range(9)])
                obs, tasks, returns, masks, actions, values, neglogpacs, latents, epinfos = bs
                traj_ax.plot(obs[:, 0], obs[:, 1], color=colormap(i * 1. / nsamples), zorder=2, linewidth=.5, marker='o', markersize=1)

            onehot = np.zeros(ntasks)
            onehot[task] = 1
            embedding_mean, embedding_std = tuple(self.model.act_model.embedding_params([onehot]))
            print(embedding_mean, embedding_std)

            fig.add_subplot(traj_ax)

            gs10 = gridspec.GridSpecFromSubplotSpec(latent_dim, 1, subplot_spec=gs00[1:])
            for li in range(latent_dim):
                latent_ax = plt.Subplot(fig, gs10[li])
                latent_ax.set_ylabel("latent[%i]" % li)
                latent_ax.grid()
                if task > 0:
                    latent_ax.get_shared_x_axes().join(latent_ax, latent_axes[li])

                mu, sigma = embedding_mean[0][li], embedding_std[0][li]
                xs = np.linspace(mu - 2. * sigma, mu + 2 * sigma, nbins)
                ys = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (xs - mu)) ** 2))
                latent_ax.plot(xs, ys, linewidth=1.5, color="black")

                latent_ax.fill_between(xs, ys, np.zeros_like(ys), facecolor="lightgrey", interpolate=True)

                for i, batch in enumerate(task_data[task]):
                    bs = tuple([np.array([batch[i][k] for i in range(len(batch))]) for k in range(9)])
                    obs, tasks, returns, masks, actions, values, neglogpacs, latents, epinfos = bs
                    latent_ax.axvline(latents[0, li], zorder=2, linewidth=2, color=colormap(i * 1. / nsamples))

                latent_axes.append(latent_ax)
                fig.add_subplot(latent_ax)

        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(osp.join(self.plot_folder, 'embed_%05d.png' % update))
        # plt.clf()
