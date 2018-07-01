import os
import sys
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf

import point_env
from inference_net import InferenceNetwork
from model import Model
from point_env import PointEnv
from policies import MlpEmbedPolicy
from sampler import Sampler
from visualizer import Visualizer

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))

from baselines import logger
from collections import deque
from baselines.common import explained_variance
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize


def constfn(val):
    def f(_):
        return val

    return f


def safemean(xs):
    if len(xs) == 0:
        return np.nan
    return np.nan if len(xs) == 0 else np.mean(xs)


def learn(*, policy, env, task_space, latent_space, traj_size,
                               nbatches, total_timesteps,
          policy_entropy, embedding_entropy, inference_coef, inference_horizon, lr,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, pi_opt_epochs=4, inference_opt_epochs=4, cliprange=0.2, seed=None,
          save_interval=0, load_path=None, plot_interval=50, plot_folder=None, log_folder=None):
    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)

    if plot_folder and not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    # nbatch = nenvs * nsteps
    # nbatch_train = nbatch // nminibatches

    make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space,
                               task_space=task_space, latent_space=latent_space,
                               # nbatch_act=nenvs,
                               # nbatch_train=nbatch_train,
                               traj_size=traj_size,
                               policy_entropy=policy_entropy,
                               embedding_entropy=embedding_entropy,
                               inference_horizon=inference_horizon,
                               vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm, seed=seed)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    if load_path is not None:
        model.load(load_path)

    sampler = Sampler(env=env, model=model, traj_size=traj_size, inference_opt_epochs=inference_opt_epochs,
                      inference_coef=inference_coef, gamma=gamma, lam=lam)

    visualizer = Visualizer(model, env, plot_folder)

    ntasks = len(point_env.TASKS)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps // traj_size
    for update in range(1, nupdates + 1):
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        inference_losses = []
        sampled_tasks = []
        completion_ratios = []
        neglogpacs_total = []

        training_batches = []
        visualization_batches = []
        act_latents = []
        for i_batch in range(nbatches):
            for task in range(ntasks):
                obs, returns, masks, actions, values, neglogpacs, latents, tasks, states, epinfos, \
                    completion_ratio, inference_loss, inference_log_likelihoods, inference_discounted_log_likelihoods, \
                    inference_means, inference_stds = sampler.run(task)
                epinfobuf.extend(epinfos)
                training_batches.append((obs, tasks, returns, masks, actions, values, neglogpacs, states))
                visualization_batches.append((obs, tasks, returns, masks, actions, values, neglogpacs, latents, epinfos,
                                              inference_means, inference_stds))
                neglogpacs_total .append(neglogpacs)
                act_latents.append(latents)
                inference_losses.append(inference_loss)
                sampled_tasks.append(tasks)
                completion_ratios.append(completion_ratio)

        train_return = model.train(lrnow, cliprangenow, training_batches)
        train_latents = train_return[-2]
        advantages = train_return[-1]
        latent_distances = np.abs(np.array(act_latents) - np.array(train_latents))
        mblossvals = train_return[:-2]

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
        if update % log_interval == 0 or update == 1:
            # ev = explained_variance(values, returns)
            # logger.logkv("explained_variance", float(ev))
            # logger.logkv("nupdates", update)
            # logger.logkv("total_timesteps", update * nbatch)
            logger.logkv("fps", fps)
            for t in range(ntasks):
                logger.logkv("completion_ratio/task%i" % t, safemean(completion_ratios[t::ntasks]))
            logger.logkv("latent_sample_error", latent_distances)
            logger.logkv("episode/reward", safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv("episode/length", safemean([epinfo['l'] for epinfo in epinfobuf]))
            for t in range(task_space.shape[0]):
                logger.logkv("sampled_task/%i" % t, np.sum(sampled_tasks[:, t]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv("loss/%s" % lossname, lossval)
            logger.logkv("loss/inference_net", safemean(inference_losses))

            logger.logkv("ppo_internals/neglogpac", safemean(neglogpacs_total))
            logger.logkv("ppo_internals/returns", safemean([b[2] for b in training_batches]))
            logger.logkv("ppo_internals/values", safemean([b[5] for b in training_batches]))
            logger.logkv("ppo_internals/advantages", safemean(advantages))

            logger.dumpkvs()
            if update == 1 and log_folder is not None:
                # save graph to visualize in TensorBoard
                writer = tf.summary.FileWriter(logdir=log_folder, graph=tf.get_default_graph())
                writer.add_graph(tf.get_default_graph())
                writer.flush()
        if plot_interval and (update % plot_interval == 0 or update == 1):
            visualizer.visualize(update, visualization_batches)
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)
    env.close()
    return model
