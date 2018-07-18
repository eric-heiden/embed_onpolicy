import copy
from typing import Callable

import numpy as np

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from matplotlib.ticker import MaxNLocator

from curriculum import BasicCurriculum, ReverseCurriculum
from ppo2embed import Model

import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os.path as osp


class Visualizer(object):
    def __init__(self, model: Model, env: DummyVecEnv, plot_folder: str, traj_plot_fn: Callable[..., None]):
        self.model = model
        self.env = env
        self.plot_folder = plot_folder
        self.traj_plot_fn = traj_plot_fn

    def visualize(self, update: int, batches, curriculum: BasicCurriculum = None):
        print("### Visualizing iteration %i ###" % update)
        ntasks = self.model.task_space.shape[0]
        nbatches = len(batches) // ntasks
        nbins = 100
        nactions = self.env.action_space.shape[0]
        latent_dim = self.model.latent_space.shape[0]
        batch_tuple_size = 11
        colormap = lambda x: matplotlib.cm.get_cmap("rainbow")(1. - x)

        # task_data_prior = [batches[t::ntasks] for t in range(ntasks)]  # batches grouped by task id
        task_data = [[] for _ in range(ntasks)]
        for task in range(ntasks):
            for batch in range(nbatches):
                task_data[task].append(batches[task * nbatches + batch])
            # task_samples = []
            # for i, batch in enumerate(task_data_prior[task]):
            #     for t in range(len(batch[0])):
            #         sample = [batch[k][t] for k in range(len(batch))]
            #         if sample[-3]["l"] == 1 and len(task_samples) > 0:  # epinfo indicates first step
            #             task_data[task].append(copy.copy(task_samples))
            #             task_samples = [sample]
            #         else:
            #             task_samples.append(sample)
            #     if len(task_samples) > 0:
            #         task_data[task].append(copy.copy(task_samples))

        title = 'Iteration %i' % update
        title += ' (%s distribution)' % ('Normal', 'Beta')[int(self.model.use_beta)]
        title += ' { ' + ', '.join("%s: %f" % (key, value) for key, value in self.model.parameters.items()) + ' }'

        additional_width = 0
        if curriculum is not None:
            additional_width = 5
        fig = plt.figure(figsize=(latent_dim * 6 + 4 + nactions * 4 + additional_width, ntasks * 3))
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

            plot_index = 0

            curriculum_plots = 0
            if curriculum is not None:
                curriculum_plots = 1

            # plot trajectories
            gs00 = gridspec.GridSpecFromSubplotSpec(1, 2 + latent_dim * 2 + nactions + curriculum_plots, subplot_spec=gs0[task])

            if self.traj_plot_fn is not None:
                self.traj_plot_fn(fig, gs00[plot_index], task, batch_tuple_size, task_data[task], colormap)
            plot_index += 1

            embedding_mean, embedding_std = embedding_means[task], embedding_stds[task]

            # plot actions
            for da in range(self.env.action_space.shape[0]):
                action_ax = plt.Subplot(fig, gs00[plot_index])
                action_ax.set_title("action[%i]" % da)
                action_ax.grid()
                if da > 0 or task > 0:
                    action_ax.get_shared_x_axes().join(action_ax, action_axes[0])
                    action_ax.get_shared_y_axes().join(action_ax, action_axes[0])

                action_ax.axhline(self.env.action_space.low[da], zorder=2, linewidth=2, color="r")
                action_ax.axhline(self.env.action_space.high[da], zorder=2, linewidth=2, color="r")

                for i, batch in enumerate(task_data[task]):
                    # bs = tuple([np.array([batch[i][k] for i in range(len(batch))]) for k in range(batch_tuple_size)])
                    obs, tasks, returns, masks, actions, values, neglogpacs, latents, epinfos, \
                        inference_means, inference_stds = tuple(batch)
                    action_ax.plot(actions[:, da], '.-', zorder=2, color=colormap(i * 1. / nsamples))

                action_axes.append(action_ax)
                fig.add_subplot(action_ax)
                plot_index += 1

            # plot values / returns
            value_ax = plt.Subplot(fig, gs00[plot_index])
            value_ax.set_title("values and returns")
            value_ax.grid()
            if task > 0:
                value_ax.get_shared_x_axes().join(value_ax, value_axes[0])
                value_ax.get_shared_y_axes().join(value_ax, value_axes[0])

            for i, batch in enumerate(task_data[task]):
                # bs = tuple([np.array([batch[i][k] for i in range(len(batch))]) for k in range(batch_tuple_size)])
                obs, tasks, returns, masks, actions, values, neglogpacs, latents, epinfos, \
                    inference_means, inference_stds = tuple(batch)
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
            plot_index += 1

            # plot curriculum
            curriculum_ax = plt.Subplot(fig, gs00[plot_index])
            curriculum_ax.set_title("curriculum progress")

            if isinstance(curriculum, ReverseCurriculum):
                curriculum_ax.set_ylim([0, curriculum.max_progress])
                curriculum_ax.set_xlim([0, nbatches])
                for i, p in enumerate(curriculum.progress[task]):
                    curriculum_ax.bar([i+.5], [p], color=colormap(i * 1. / nsamples))
                    curriculum_ax.bar([i+.5], [curriculum.max_progress - p], bottom=[p], color="lightgrey")
                curriculum_ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            fig.add_subplot(curriculum_ax)
            plot_index += 1

            # plot embeddings
            gs10 = gridspec.GridSpecFromSubplotSpec(1, latent_dim,
                                                    subplot_spec=gs00[plot_index:plot_index + latent_dim])
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
                    # bs = tuple([np.array([batch[i][k] for i in range(len(batch))]) for k in range(batch_tuple_size)])
                    obs, tasks, returns, masks, actions, values, neglogpacs, latents, epinfos, \
                        inference_means, inference_stds = tuple(batch)
                    latent_ax.axvline(latents[0, li], zorder=2, linewidth=2, color=colormap(i * 1. / nsamples))

                latent_axes.append(latent_ax)
                fig.add_subplot(latent_ax)
            plot_index += latent_dim

            # plot inference performance
            gs20 = gridspec.GridSpecFromSubplotSpec(1, latent_dim, subplot_spec=gs00[plot_index:plot_index + latent_dim])

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
                                      np.ones_like(xs) * (true_mu + 2.5 * true_sigma),
                                      np.ones_like(xs) * (true_mu - 2.5 * true_sigma),
                                      facecolor="lightgrey",
                                      alpha=.6,
                                      zorder=1)
                infer_ax.plot(xs, np.ones_like(xs) * true_mu, color="black", zorder=2)

                for i, batch in enumerate(task_data[task]):
                    # bs = tuple([np.array([batch[i][k] for i in range(len(batch))]) for k in range(batch_tuple_size)])
                    obs, tasks, returns, masks, actions, values, neglogpacs, latents, epinfos, \
                        inference_means, inference_stds = tuple(batch)
                    mus = inference_means[:, li]
                    sigmas = inference_stds[:, li]
                    xs = np.arange(0, len(obs))
                    # infer_ax.fill_between(xs,
                    #                       mus + sigmas,
                    #                       mus - sigmas,
                    #                       facecolor=colormap(i * 1. / nsamples),
                    #                       alpha=.1,
                    #                       zorder=1)
                    infer_ax.plot(xs, mus, color=colormap(i * 1. / nsamples), marker='o', markersize=1.5, zorder=2)
                    # if i > 3:
                    #     break
                # if task == 0 and li == 0:
                #     infer_ax.legend()
                infer_axes.append(infer_ax)
                fig.add_subplot(infer_ax)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(osp.join(self.plot_folder, 'embed_%05d.png' % update))
        plot_index += latent_dim

        expand = True
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        shape = (nrows, ncols, 3) if not expand else (-1, nrows, ncols, 3)
        return np.fromstring(buf, dtype=np.uint8).reshape(shape)
        # plt.clf()
