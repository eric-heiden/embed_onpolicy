import os
import os.path as osp
import sys
import time
from collections import deque
from typing import Union, Callable

import imageio
import numpy as np
import tensorflow as tf

from curriculum import BasicCurriculum, ReverseCurriculum
from model import Model
from sampler import Sampler
from visualizer import Visualizer

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))

from baselines import logger
from baselines.common.math_util import explained_variance


def const_fn(val):
    return lambda _: val


def linear_transition(start, end, end_iteration: int, continue_beyond_end: bool = False,
                      absolute_min = None, absolute_max = None):
    if continue_beyond_end:
        if absolute_min:
            return lambda iteration: max(absolute_min, (end - start) * iteration / end_iteration + start)
        if absolute_max:
            return lambda iteration: min(absolute_max, (end - start) * iteration / end_iteration + start)
        return lambda iteration: (end - start) * iteration / end_iteration + start
    return lambda iteration: (end - start) * min(1., iteration / end_iteration) + start


TimeVarying = Union[float, Callable[[int], float]]


def safemean(xs):
    if len(xs) == 0:
        return np.nan
    return np.nan if len(xs) == 0 else np.mean(xs)


def make_callable(what: TimeVarying):
    if isinstance(what, float):
        what = const_fn(what)
    else:
        assert callable(what)
    return what


def learn(*, policy, env_fn, unwrap_env, task_space, latent_space, traj_size,
          nbatches, total_timesteps,
          policy_entropy: TimeVarying, embedding_entropy: TimeVarying, inference_coef: TimeVarying,
          inference_horizon, lr: TimeVarying,
          vf_coef: TimeVarying =0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, inference_opt_epochs=4, cliprange: TimeVarying = 0.2, seed=None,
          save_interval=50, load_path=None, plot_interval=50, plot_event_interval=200,
          plot_folder=None, traj_plot_fn=None, log_folder=None,
          render_interval=-1, render_fn=None, render_fps=20, curriculum_fn=BasicCurriculum,
          use_embedding=True,
          **kwargs):
    lr = make_callable(lr)
    cliprange = make_callable(cliprange)
    policy_entropy = make_callable(policy_entropy)
    embedding_entropy = make_callable(embedding_entropy)
    vf_coef = make_callable(vf_coef)
    inference_coef = make_callable(inference_coef)

    total_timesteps = int(total_timesteps)

    env = env_fn(task=0)  # instantiate and environment ust to get the spaces
    nenvs = 1  # env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    ntasks = task_space.shape[0]

    make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space,
                               task_space=task_space, latent_space=latent_space,
                               traj_size=traj_size, cliprange=cliprange, lr=lr,
                               policy_entropy=policy_entropy,
                               embedding_entropy=embedding_entropy,
                               inference_horizon=inference_horizon,
                               vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm, seed=seed,
                               use_embedding=use_embedding,
                               **kwargs)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'configuration.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps({
                "make_model": make_model,
                "make_env": env_fn,
                "render_fn": render_fn,
                "traj_plot_fn": traj_plot_fn,
                "unwrap_env_fn": unwrap_env,
                "traj_size": traj_size,
                "task_space": task_space,
                "latent_space": latent_space,
                "curriculum_fn": curriculum_fn,
                "seed": seed,
                "gamma": gamma,
                "lambda": lam,
                "vf_coef": vf_coef,
                "policy_entropy": policy_entropy,
                "embedding_entropy": embedding_entropy,
                "inference_horizon": inference_horizon,
                "max_grad_norm": max_grad_norm,
                "nbatches": nbatches,
                "total_timesteps": total_timesteps,
                "cliprange": cliprange,
                "lr": lr,
                "plot_folder": plot_folder,
                "use_embedding": use_embedding
            }))
    model = make_model()
    if load_path is not None:
        model.load(load_path)

    sampler = Sampler(env=env, unwrap_env=unwrap_env, model=model, traj_size=traj_size,
                      inference_opt_epochs=inference_opt_epochs,
                      inference_coef=inference_coef,
                      gamma=gamma, lam=lam,
                      use_embedding=use_embedding)

    curriculum = curriculum_fn(env_fn, unwrap_env=unwrap_env, batches=nbatches, tasks=ntasks)

    visualizer = Visualizer(model, env, unwrap_env, plot_folder, traj_plot_fn, use_embedding=use_embedding)

    if render_fn is not None and render_interval > 0:
        env.render()

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    summary_writer = tf.summary.FileWriter(logdir=osp.join(log_folder, "tb"), graph=tf.get_default_graph())
    vis_placeholder = None
    vis_summary = None

    def log(key, value, update):
        logger.logkv(key, value)
        summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=key,
                                                                      simple_value=value)]), update)

    nupdates = total_timesteps // traj_size
    for iteration in range(nupdates):
        tstart = time.time()

        inference_losses = []
        sampled_tasks = []
        completion_ratios = np.zeros(ntasks)
        neglogpacs_total = []
        explained_variances = []

        training_batches = []
        visualization_batches = []
        act_latents = []
        video = []

        for task, envs in enumerate(curriculum.strategy):
            for i_batch, env in enumerate(envs):
                if render_fn is not None and render_interval > 0 and (
                        iteration == 0 or iteration % render_interval == 0) and i_batch == 0:
                    rf = render_fn(task, iteration)
                    obs, returns, masks, actions, values, neglogpacs, latents, tasks, states, epinfos, \
                    completions, inference_loss, inference_log_likelihoods, inference_discounted_log_likelihoods, \
                    sampled_video, extras = sampler.run(iteration, env, task, render=rf)
                    video += sampled_video
                else:
                    obs, returns, masks, actions, values, neglogpacs, latents, tasks, states, epinfos, \
                    completions, inference_loss, inference_log_likelihoods, inference_discounted_log_likelihoods, \
                    extras = sampler.run(iteration, env, task)
                epinfobuf.extend(epinfos)
                training_batches.append((obs, tasks, returns, masks, actions, values, neglogpacs, states))
                visualization_batches.append((obs, tasks, returns, masks, actions, values, neglogpacs, latents, epinfos, extras))
                neglogpacs_total.append(neglogpacs)
                act_latents.append(latents)
                inference_losses.append(inference_loss)
                sampled_tasks.append(tasks)
                explained_variances.append(explained_variance(values, returns))
                completion_ratios[task] += completions
        completion_ratios /= 1. * nbatches

        if len(video) > 0 and plot_folder is not None:
            imageio.mimsave(osp.join(plot_folder, 'embed_%05d.mp4' % iteration), video, fps=render_fps)

        train_return = model.train(iteration, training_batches)
        train_latents = train_return[-2]
        advantages = train_return[-1]
        latent_distances = [np.mean(np.abs(np.array(al) - np.array(tl))) for al, tl in zip(act_latents, train_latents)]
        mblossvals = train_return[:-2]

        curriculum.update(visualization_batches)

        # if states is None:  # nonrecurrent version
        #     inds = np.arange(traj_size)
        #     for _ in range(pi_opt_epochs):
        #         # np.random.shuffle(inds)
        #         for start in range(0, nbatch, nbatch_train):
        #             end = start + nbatch_train
        #             mbinds = inds[start:end]
        #             # print("mbinds", mbinds)
        #             # print("tasks", tasks.shape)
        #
        #             # train inference network
        #             # slices = (arr[mbinds] for arr in (obs, actions, latents))
        #             # inference_losses.append(model.inference_model.train(*slices))
        #
        #             # train policy and embedding network
        #             slices = (arr[mbinds] for arr in (obs, tasks, returns, masks, actions, values, neglogpacs, latents))
        #             train_return = model.train(lrnow, cliprangenow, *slices, inference_loss=inference_loss)
        #             train_latents = train_return[-1]
        #             latent_distances.append(list(np.abs(latents[mbinds] - train_latents)))
        #             mblossvals.append(train_return[:-1])
        # else:  # recurrent version
        #     assert nenvs % nminibatches == 0
        #     envsperbatch = nenvs // nminibatches
        #     envinds = np.arange(nenvs)
        #     flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
        #     envsperbatch = nbatch_train // nsteps
        #     for _ in range(pi_opt_epochs):
        #         np.random.shuffle(envinds)
        #         for start in range(0, nenvs, envsperbatch):
        #             end = start + envsperbatch
        #             mbenvinds = envinds[start:end]
        #             mbflatinds = flatinds[mbenvinds].ravel()
        #             slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
        #             mbstates = states[mbenvinds]
        #             mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        latent_distances = np.mean(latent_distances)

        sampled_tasks = np.array(sampled_tasks)

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(traj_size / (tnow - tstart))
        if plot_interval and (iteration % plot_interval == 0 or iteration == 0):
            image = visualizer.visualize(iteration, visualization_batches, curriculum)
            if iteration == 0:
                vis_placeholder = tf.placeholder(tf.uint8, image.shape)
                vis_summary = tf.summary.image('episode_microscope', vis_placeholder)
            if iteration % plot_event_interval == 0 or iteration == 0:
                summary_writer.add_summary(vis_summary.eval(feed_dict={vis_placeholder: image}), iteration)
        if iteration % log_interval == 0 or iteration == 0:
            with tf.name_scope('summaries'):
                log("explained_variance", safemean(explained_variances), iteration)
                log("fps", fps, iteration)
                for t in range(ntasks):
                    log("completion_ratio/task_%i" % t, safemean(completion_ratios[t::ntasks]), iteration)
                if isinstance(curriculum, ReverseCurriculum):
                    for t in range(ntasks):
                        log("curriculum_progress/task_%i" % t, curriculum.task_progress_ratios[t], iteration)
                log("latent_sample_error", latent_distances, iteration)
                log("episode/reward", safemean([epinfo['r'] for epinfo in epinfobuf]), iteration)
                log("episode/length", safemean([epinfo['l'] for epinfo in epinfobuf]), iteration)
                # for t in range(task_space.shape[0]):
                #     log("sampled_task/%i" % t, np.sum(sampled_tasks[:, t]), update)
                log('time_elapsed', tnow - tfirststart, iteration)
                log('iteration', iteration, iteration)
                for (lossval, lossname) in zip(lossvals, model.loss_names):
                    log("loss/%s" % lossname, lossval, iteration)
                log("loss/inference_net", safemean(inference_losses), iteration)

                log("params/lr", lr(iteration), iteration)
                log("params/cliprange", cliprange(iteration), iteration)
                log("params/vf_coef", vf_coef(iteration), iteration)
                if use_embedding:
                    log("params/inference_coef", inference_coef(iteration), iteration)
                    log("params/embedding_entropy", embedding_entropy(iteration), iteration)
                log("params/policy_entropy", policy_entropy(iteration), iteration)

                # log("ppo_internals/neglogpac", safemean(neglogpacs_total), update)
                # log("ppo_internals/returns", safemean([b[2] for b in training_batches]), update)
                # log("ppo_internals/values", safemean([b[5] for b in training_batches]), update)
                # log("ppo_internals/advantages", safemean(advantages), update)

                print("Run:", log_folder)
                logger.dumpkvs()
                summary_writer.flush()
        if save_interval and (iteration % save_interval == 0 or iteration == 0) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i' % iteration)
            print('Saving to', savepath)
            model.save(savepath)
    env.close()
    return model
