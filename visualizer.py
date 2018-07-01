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
        nactions = self.env.action_space.shape[0]
        latent_dim = self.model.latent_space.shape[0]
        batch_tuple_size = 11
        colormap = lambda x: matplotlib.cm.get_cmap("winter")(1. - x)

        task_data_prior = [batches[t::ntasks] for t in range(ntasks)]  # batches grouped by task id
        task_data = [[] for _ in range(ntasks)]
        for task in range(ntasks):
            task_samples = []
            for i, batch in enumerate(task_data_prior[task]):
                for t in range(len(batch[0])):
                    sample = [batch[k][t] for k in range(len(batch))]
                    if sample[-3]["l"] == 1 and len(task_samples) > 0:  # epinfo indicates first step
                        task_data[task].append(copy.copy(task_samples))
                        task_samples = [sample]
                    else:
                        task_samples.append(sample)
                if len(task_samples) > 0:
                    task_data[task].append(copy.copy(task_samples))

        title = 'Iteration %i' % update
        title += ' (%s distribution)' % ('Normal', 'Beta')[int(self.model.use_beta)]
        title += ' { ' + ', '.join("%s: %f" % (key, value) for key, value in self.model.parameters.items()) + ' }'

        fig = plt.figure(figsize=(latent_dim * 6 + 2 + nactions * 2, ntasks * 3))
        fig.suptitle(title)

        gs0 = gridspec.GridSpec(ntasks, 1)

        latent_axes = []
        action_axes = []
        value_axes = []
        infer_axes = []

        embedding_means, embedding_stds = [], []
        latent_mins = np.zeros(latent_dim)
        latent_maxs = np.zeros(latent_dim)
        for task in range(ntasks):
            onehot = np.zeros(ntasks)
            onehot[task] = 1
            embedding_mean, embedding_std = tuple(self.model.act_model.embedding_params([onehot]))
            embedding_means.append(embedding_mean[0])
            embedding_stds.append(embedding_std[0])
        for d in range(latent_dim):
            latent_mins[d] = np.min([embedding_means[t][d] - 3. * embedding_stds[t][d] for t in range(ntasks)])
            latent_maxs[d] = np.max([embedding_means[t][d] + 3. * embedding_stds[t][d] for t in range(ntasks)])

        for task in range(ntasks):
            nsamples = float(len(task_data[task]))

            # plot trajectories
            # print("Plotting task", task)
            gs00 = gridspec.GridSpecFromSubplotSpec(1, 2 + latent_dim * 2 + nactions, subplot_spec=gs0[task])
            traj_ax = plt.Subplot(fig, gs00[0])
            traj_ax.set_title("Task %i" % (task + 1))
            traj_ax.grid()
            traj_ax.set_xlim([-5, 5])
            traj_ax.set_ylim([-5, 5])
            traj_ax.set_aspect('equal')
            goal = plt.Circle(point_env.TASKS[task], radius=point_env.MIN_DIST, color='orange')
            traj_ax.add_patch(goal)
            for tl in traj_ax.get_xticklabels() + traj_ax.get_yticklabels():
                tl.set_visible(False)

            for i, batch in enumerate(task_data[task]):
                bs = tuple([np.array([batch[i][k] for i in range(len(batch))]) for k in range(batch_tuple_size)])
                obs, tasks, returns, masks, actions, values, neglogpacs, latents, epinfos, \
                    inference_means, inference_stds = bs
                traj_ax.plot([0] + obs[:, 0], [0] + obs[:, 1], color=colormap(i * 1. / nsamples),
                             zorder=2, linewidth=.5, marker='o', markersize=1)
            fig.add_subplot(traj_ax)

            embedding_mean, embedding_std = embedding_means[task], embedding_stds[task]

            # plot actions
            for da in range(self.env.action_space.shape[0]):
                action_ax = plt.Subplot(fig, gs00[1 + da])
                action_ax.set_title("action[%i]" % da)
                action_ax.grid()
                if da > 0 or task > 0:
                    action_ax.get_shared_x_axes().join(action_ax, action_axes[0])
                    action_ax.get_shared_y_axes().join(action_ax, action_axes[0])

                action_ax.axhline(self.env.action_space.low[da], zorder=2, linewidth=2, color="r")
                action_ax.axhline(self.env.action_space.high[da], zorder=2, linewidth=2, color="r")

                for i, batch in enumerate(task_data[task]):
                    bs = tuple([np.array([batch[i][k] for i in range(len(batch))]) for k in range(batch_tuple_size)])
                    obs, tasks, returns, masks, actions, values, neglogpacs, latents, epinfos, \
                        inference_means, inference_stds = bs
                    action_ax.plot(actions[:, da], '.-', zorder=2, color=colormap(i * 1. / nsamples))

                action_axes.append(action_ax)
                fig.add_subplot(action_ax)

            # plot values / returns
            value_ax = plt.Subplot(fig, gs00[1 + nactions])
            value_ax.set_title("values and returns")
            value_ax.grid()
            if task > 0:
                value_ax.get_shared_x_axes().join(value_ax, value_axes[0])
                value_ax.get_shared_y_axes().join(value_ax, value_axes[0])

            for i, batch in enumerate(task_data[task]):
                bs = tuple([np.array([batch[i][k] for i in range(len(batch))]) for k in range(batch_tuple_size)])
                obs, tasks, returns, masks, actions, values, neglogpacs, latents, epinfos, \
                    inference_means, inference_stds = bs
                xs = np.arange(0, len(obs))
                value_ax.fill_between(xs,
                                      returns,
                                      values,
                                      facecolor="orange",
                                      alpha=.2,
                                      zorder=1)
                if i == len(task_data[task]) - 1:
                    value_ax.plot(returns, zorder=2, color=colormap(i * 1. / nsamples), label="returns")
                    value_ax.plot(values, '--', zorder=2, color=colormap(i * 1. / nsamples), label="values")
                else:
                    value_ax.plot(returns, zorder=2, color=colormap(i * 1. / nsamples))
                    value_ax.plot(values, '--', zorder=2, color=colormap(i * 1. / nsamples))
                # break  # TODO remove?
            if task == 0:
                value_ax.legend()

            value_axes.append(value_ax)
            fig.add_subplot(value_ax)

            # plot embeddings
            gs10 = gridspec.GridSpecFromSubplotSpec(1, latent_dim, subplot_spec=gs00[2 + nactions:2 + latent_dim + nactions])
            for li in range(latent_dim):
                latent_ax = plt.Subplot(fig, gs10[li])
                latent_ax.set_title("latent[%i]" % li)
                latent_ax.grid()
                if task > 0:
                    latent_ax.get_shared_x_axes().join(latent_ax, latent_axes[li])

                mu, sigma = embedding_mean[li], embedding_std[li]
                xs = np.linspace(latent_mins[li], latent_maxs[li], nbins)
                ys = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (xs - mu)) ** 2))
                latent_ax.plot(xs, ys, linewidth=1.5, color="black")

                latent_ax.fill_between(xs, ys, np.zeros_like(ys), facecolor="lightgrey", interpolate=True)

                for i, batch in enumerate(task_data[task]):
                    bs = tuple([np.array([batch[i][k] for i in range(len(batch))]) for k in range(batch_tuple_size)])
                    obs, tasks, returns, masks, actions, values, neglogpacs, latents, epinfos, \
                        inference_means, inference_stds = bs
                    latent_ax.axvline(latents[0, li], zorder=2, linewidth=2, color=colormap(i * 1. / nsamples))

                latent_axes.append(latent_ax)
                fig.add_subplot(latent_ax)

            # plot inference performance
            gs20 = gridspec.GridSpecFromSubplotSpec(1, latent_dim, subplot_spec=gs00[-latent_dim:])

            for li in range(latent_dim):
                infer_ax = plt.Subplot(fig, gs20[li])
                infer_ax.set_title("inferred latent[%i]" % li)
                infer_ax.grid()
                if task > 0:
                    infer_ax.get_shared_x_axes().join(infer_ax, infer_axes[li])
                    infer_ax.get_shared_y_axes().join(infer_ax, infer_axes[li])

                true_mu, true_sigma = embedding_mean[li], embedding_std[li]
                xs = np.arange(0, self.model.traj_size)
                infer_ax.fill_between(xs,
                                      np.ones_like(xs) * (true_mu + true_sigma),
                                      np.ones_like(xs) * (true_mu - true_sigma),
                                      facecolor="r",
                                      alpha=.1,
                                      zorder=1)
                infer_ax.plot(xs, np.ones_like(xs) * true_mu, color="r", label="true", zorder=2)

                for i, batch in enumerate(task_data[task]):
                    bs = tuple([np.array([batch[i][k] for i in range(len(batch))]) for k in range(batch_tuple_size)])
                    obs, tasks, returns, masks, actions, values, neglogpacs, latents, epinfos, \
                        inference_means, inference_stds = bs
                    mus = inference_means[:, li]
                    sigmas = inference_means[:, li]
                    xs = np.arange(0, len(obs))
                    infer_ax.fill_between(xs,
                                          mus + sigmas,
                                          mus - sigmas,
                                          facecolor=colormap(i * 1. / nsamples),
                                          alpha=.1,
                                          zorder=1)
                    infer_ax.plot(xs, mus, color=colormap(i * 1. / nsamples), marker='o', markersize=1.5, zorder=2)
                    if i > 3:
                        break
                if task == 0 and li == 0:
                    infer_ax.legend()
                infer_axes.append(infer_ax)
                fig.add_subplot(infer_ax)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(osp.join(self.plot_folder, 'embed_%05d.png' % update))
        # plt.clf()
