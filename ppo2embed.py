import os
import sys
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf

import point_env
from inference_net import InferenceNetwork
from point_env import PointEnv
from policies import MlpEmbedPolicy

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))

from baselines import logger
from collections import deque
from baselines.common import explained_variance
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, task_space, latent_space, traj_size,
                 policy_entropy, vf_coef, max_grad_norm, embedding_entropy=0., inference_coef=0.1, inference_horizon=5, seed=None):
        sess = tf.get_default_session()
        with tf.variable_scope("PPO"):
            act_model = policy(sess, ob_space, ac_space, task_space, latent_space, traj_size=1,
                               reuse=False, seed=seed, name="model")  # type: MlpEmbedPolicy
            train_model = policy(sess, ob_space, ac_space, task_space, latent_space, traj_size=traj_size,
                                 reuse=True, seed=seed, name="model")  # type: MlpEmbedPolicy
            inference_model = InferenceNetwork(sess, ob_space, ac_space, latent_space, horizon=inference_horizon)

            A = tf.placeholder(dtype=tf.float32, shape=train_model.pd.batch_shape, name="actions")
            # A = train_model.pd.sample(name="A")
            ADV = tf.placeholder(tf.float32, [None], name="advantages")
            # ADV2 = tf.stack((ADV, ADV), axis=1, name="stacked_ADV")
            R = tf.placeholder(tf.float32, [None], name="returns")
            OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], name="old_neglogpac")
            OLDVPRED = tf.placeholder(tf.float32, [None], name="old_vpred")
            LR = tf.placeholder(tf.float32, [], name="learning_rate")
            CLIPRANGE = tf.placeholder(tf.float32, [], name="clip_range")

            # neglogpac = train_model.pd.neglogp(A)
            neglogpac = train_model.neg_log_prob(A, "neglogpac")
            entropy = tf.reduce_mean(train_model.pd.entropy(), name="entropy")

            with tf.name_scope("ValueFunction"):
                vpred = train_model.vf
                vpredclipped = tf.identity(OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, -CLIPRANGE, CLIPRANGE,
                                                                       name="clip_vf"), name="vpred_clipped")
                vf_losses1 = tf.square(vpred - R, name="vf_loss1")
                vf_losses2 = tf.square(vpredclipped - R, name="vf_loss2")
                vf_loss = tf.identity(.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2)), name="vf_loss")

            with tf.name_scope("PolicyGradient"):
                ratio = tf.exp(OLDNEGLOGPAC - neglogpac, name="nlp_ratio")
                pg_losses = tf.identity(-ADV * ratio, name="pg_loss1")
                pg_losses2 = tf.identity(-ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE), name="pg_loss2")
                pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2), name="pg_loss")
                approxkl = tf.identity(.5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC)), name="approx_kl")
                clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)), name="clip_frac")
                loss = tf.identity(pg_loss + vf_loss * vf_coef, name="policy_loss")

                _inference_loss = tf.placeholder(tf.float32, [], name="inference_loss")

                final_loss = tf.identity(loss
                                         - policy_entropy * entropy
                                         - embedding_entropy * train_model.embedding_entropy,
                                         # + inference_coef * _inference_loss,
                             name="final_loss")

            with tf.variable_scope('model', reuse=True):
                params = tf.trainable_variables()
                print("TRAINABLE VARS", params)

            with tf.name_scope("Training"):
                grads = tf.gradients(final_loss, params)
                if max_grad_norm is not None:
                    grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm, name="clipped_grads")
                grads = list(zip(grads, params))
                trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5, name="adam_opt")
                _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, obs, tasks, returns, masks, actions, values, neglogpacs, latents, inference_loss, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {
                train_model.Observation: obs,
                train_model.Task: [tasks[0]],
                A: actions,
                ADV: advs,
                R: returns,
                LR: lr,
                CLIPRANGE: cliprange,
                OLDNEGLOGPAC: neglogpacs,
                OLDVPRED: values,
                _inference_loss: inference_loss,
                # train_model.Embedding: latents
            }
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, approxkl, clipfrac, entropy, train_model.embedding_entropy, final_loss, train_model.Embedding, _train],
                td_map
            )[:-1]

        def get_latent(task: int):
            one_hot = np.zeros(act_model.Task.shape)
            one_hot[:, task] = 1
            latent = sess.run(act_model.Embedding, {act_model.Task: one_hot})
            return latent[0]

        self.loss_names = ['policy_loss', 'value_loss', 'approxkl', 'clipfrac', 'policy_entropy', 'embedding_entropy', 'final_loss']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model
        self.inference_model = inference_model
        self.act_model = act_model
        self.step = act_model.step
        self.step_from_task = act_model.step_from_task
        self.value = act_model.value
        self.value_from_task = act_model.value_from_task
        self.get_latent = get_latent
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        self.task_space = task_space
        self.latent_space = latent_space
        self.action_space = ac_space
        self.observation_space = ob_space

        self.use_beta = act_model.use_beta
        tf.global_variables_initializer().run(session=sess)  # pylint: disable=E1101


class Runner(object):

    def __init__(self, *, env: DummyVecEnv, model: Model, gamma, lam, traj_size: int = 20, inference_opt_epochs: int = 4):
        self.env = env
        self.model = model
        nenv = env.num_envs
        assert(nenv == 1)  # ensure to sample from embedding the same number of steps, in training and acting
        self.batch_ob_shape = (nenv * traj_size,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

        self.lam = lam
        self.gamma = gamma
        self.traj_size = traj_size
        self.inference_opt_epochs = inference_opt_epochs

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_latents, mb_tasks, mb_values, mb_dones, mb_neglogpacs = \
            [], [], [], [], [], [], [], []
        mb_states = self.states

        self.tasks = [e.select_next_task() for e in self.env.envs]
        self.onehots = []
        for task in self.tasks:
            one_hot = np.zeros((self.model.task_space.shape[0],))
            one_hot[task] = 1
            self.onehots.append(one_hot)
        self.latents = [self.model.get_latent(t) for t in self.tasks]

        epinfos = []
        completions = 0
        traj_window = deque(maxlen=self.model.inference_model.horizon)
        traj_windows = []
        discounts = []

        for step in range(self.traj_size):
            actions, values, mb_states, neglogpacs = self.model.step(self.latents, self.obs, self.states, self.dones)
            # actions, values, mb_states, neglogpacs = self.model.step_from_task(
            #     self.onehots, self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            # actions = np.clip(actions, self.model.action_space.low[0], self.model.action_space.high[0])
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            mb_rewards.append(rewards)

            mb_tasks.append(infos[-1]["episode"]["task"])
            mb_latents.append(np.array(self.latents).flatten())

            epinfos += [info["episode"] for info in infos]

            if any(self.dones):
                self.obs[:] = self.env.reset()
                # self.tasks = [e.task for e in self.env.envs]
                # self.onehots = []
                # for task in self.tasks:
                #     one_hot = np.zeros((self.model.task_space.shape[0],))
                #     one_hot[task] = 1
                #     self.onehots.append(np.copy(one_hot))
                # self.latents = [self.model.get_latent(t) for t in self.tasks]

                completions += 1
            if any(self.dones) or step == 0:
                # fill horizon buffer with step 0 copies of trajectory
                for _ in range(self.model.inference_model.horizon):
                    traj_window.append(np.concatenate((self.obs, actions)))
                discounts.append(self.gamma)
            else:
                discounts.append(discounts[-1] * self.gamma)

            traj_window.append(np.concatenate((self.obs, actions)))
            traj_windows.append(np.array(traj_window).flatten())

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_tasks = np.asarray(mb_tasks, dtype=np.float32)
        mb_latents = np.asarray(mb_latents, dtype=np.float32)
        last_values = self.model.value(self.latents, self.obs, self.states, self.dones)
        traj_windows = np.array(traj_windows)
        discounts = np.array(discounts)

        inference_loss, inference_discounted_log_likelihoods = 0, []
        # train and evaluate inference network
        for epoch in range(self.inference_opt_epochs):
            idxs = np.arange(self.traj_size)
            if epoch < self.inference_opt_epochs - 1:
                np.random.shuffle(idxs)
            # TODO shuffle the input for a better training outcome? Is this correct?!
            inference_lll = self.model.inference_model.train(traj_windows[idxs], discounts[idxs], mb_latents[idxs])
            inference_loss, inference_discounted_log_likelihoods = tuple(inference_lll)

        print("MEAN reward:", np.mean(mb_rewards))
        completion_ratio = completions * 1. / self.traj_size

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0

        for t in reversed(range(self.traj_size)):
            if t == self.traj_size - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_values + inference_discounted_log_likelihoods.reshape(mb_advs.shape)

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)), mb_latents,
                mb_tasks, mb_states, epinfos, completion_ratio, inference_loss, inference_discounted_log_likelihoods)


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val

    return f


def safemean(xs):
    if len(xs) == 0:
        return np.nan
    return np.nan if len(xs) == 0 else np.mean(xs)


def visualize(model: Model, env: DummyVecEnv, update: int, plot_folder: str):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    import matplotlib.pyplot as plt
    import pylab

    nenv = env.num_envs
    nsteps = 40
    nsamples = 10
    ntasks = len(point_env.TASKS)
    colormap = lambda x: matplotlib.cm.get_cmap("winter")(1.-x)

    fig, axes = plt.subplots(figsize=(ntasks * 4, 8), nrows=2, ncols=ntasks, sharey='row')
    fig.suptitle(('Iteration %i' % update) + (' (using %s distribution)' % ('Normal', 'Beta')[int(model.use_beta)]))
    if ntasks == 1:
        axes = [[axes[0]], [axes[1]]]
    for t in range(ntasks):
        one_hot = np.zeros((model.task_space.shape[0],))
        one_hot[t] = 1
        onehots = [one_hot]

        ax = axes[0][t]
        ax.set_title(str(one_hot))  # "Task % i" % (t + 1))
        ax.grid()
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_aspect('equal')

        goal = plt.Circle(point_env.TASKS[t], radius=point_env.MIN_DIST, color='orange')
        ax.add_patch(goal)
        # ax.scatter([point_env.TASKS[t][0]], [point_env.TASKS[t][1]], s=50, c='r')

        ax_latent = axes[1][t]
        ax_latent.grid()
        ticks = np.arange(0, model.latent_space.shape[0], 1)
        ax_latent.set_xticks(ticks)

        for sample in range(nsamples):
            obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
            obs[:] = env.reset()
            for pointEnv in env.envs:
                pointEnv._task = t  # TODO expose via setter
                pointEnv._goal = point_env.TASKS[t]
            dones = [False for _ in range(nenv)]
            positions = [np.copy(obs[0])]

            latents = [model.get_latent(t)]
            for _ in range(nsteps):
                actions, values, mb_states, neglogpacs = model.step(latents, obs, model.initial_state, dones)
                # actions, values, mb_states, neglogpacs = model.step_from_task(
                #     onehots, obs, model.initial_state, dones)
                if not model.use_beta:
                    actions *= 0.1
                # actions = np.clip(actions, model.action_space.low[0], model.action_space.high[0])
                obs[:], rewards, dones, infos = env.step(actions)
                positions.append(np.copy(obs[0]))
                if dones[-1]:
                    break
            positions = np.array(positions)
            positions = np.reshape(positions, (-1, 2))
            ax.scatter(positions[:, 0], positions[:, 1], color=colormap(sample * 1. / nsamples), s=2, zorder=2)
            ax.plot(positions[:, 0], positions[:, 1], color=colormap(sample * 1. / nsamples), zorder=2)

            # visualize latents TODO make actions dependent on these
            ax_latent.scatter(ticks, latents, color=colormap(sample * 1. / nsamples))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(osp.join(plot_folder, 'embed_%05d.png' % update))
    plt.clf()


def visualize_old(model: Model, env: DummyVecEnv, update: int, plot_folder: str):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    import matplotlib.pyplot as plt
    import pylab

    nenv = env.num_envs
    nsteps = 40
    nsamples = 10
    ntasks = len(point_env.TASKS)
    colormap = lambda x: matplotlib.cm.get_cmap("winter")(1.-x)

    fig, axes = plt.subplots(figsize=(ntasks * 4, 8), nrows=2, ncols=ntasks, sharey='row')
    fig.suptitle(('Iteration %i' % update) + (' (using %s distribution)' % ('Normal', 'Beta')[int(model.use_beta)]))
    if ntasks == 1:
        axes = [[axes[0]], [axes[1]]]
    for t in range(ntasks):
        one_hot = np.zeros((model.task_space.shape[0],))
        one_hot[t] = 1
        onehots = [one_hot]

        ax = axes[0][t]
        ax.set_title(str(one_hot))  # "Task % i" % (t + 1))
        ax.grid()
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_aspect('equal')

        goal = plt.Circle(point_env.TASKS[t], radius=point_env.MIN_DIST, color='orange')
        ax.add_patch(goal)
        # ax.scatter([point_env.TASKS[t][0]], [point_env.TASKS[t][1]], s=50, c='r')

        ax_latent = axes[1][t]
        ax_latent.grid()
        ticks = np.arange(0, model.latent_space.shape[0], 1)
        ax_latent.set_xticks(ticks)

        for sample in range(nsamples):
            obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
            obs[:] = env.reset()
            for pointEnv in env.envs:
                pointEnv._task = t  # TODO expose via setter
                pointEnv._goal = point_env.TASKS[t]
            dones = [False for _ in range(nenv)]
            positions = [np.copy(obs[0])]

            latents = [model.get_latent(t)]
            for _ in range(nsteps):
                actions, values, mb_states, neglogpacs = model.step(latents, obs, model.initial_state, dones)
                # actions, values, mb_states, neglogpacs = model.step_from_task(
                #     onehots, obs, model.initial_state, dones)
                if not model.use_beta:
                    actions *= 0.1
                # actions = np.clip(actions, model.action_space.low[0], model.action_space.high[0])
                obs[:], rewards, dones, infos = env.step(actions)
                positions.append(np.copy(obs[0]))
                if dones[-1]:
                    break
            positions = np.array(positions)
            positions = np.reshape(positions, (-1, 2))
            ax.scatter(positions[:, 0], positions[:, 1], color=colormap(sample * 1. / nsamples), s=2, zorder=2)
            ax.plot(positions[:, 0], positions[:, 1], color=colormap(sample * 1. / nsamples), zorder=2)

            # visualize latents TODO make actions dependent on these
            ax_latent.scatter(ticks, latents, color=colormap(sample * 1. / nsamples))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(osp.join(plot_folder, 'embed_%05d.png' % update))
    plt.clf()


def learn(*, policy, env, task_space, latent_space, traj_size, total_timesteps,
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
                               inference_coef=inference_coef,
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
    runner = Runner(env=env, model=model, traj_size=traj_size, inference_opt_epochs=inference_opt_epochs,
                    gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps // traj_size
    for update in range(1, nupdates + 1):
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, latents, tasks, states, epinfos, \
            completion_ratio, inference_loss, inference_discounted_log_likelihoods = runner.run()
        epinfobuf.extend(epinfos)
        mblossvals = []

        latent_distances = []
        slices = (arr for arr in (obs, tasks, returns, masks, actions, values, neglogpacs, latents))
        train_return = model.train(lrnow, cliprangenow, *slices, inference_loss=inference_loss)
        train_latents = train_return[-1]
        latent_distances.append(list(np.abs(latents - train_latents)))
        mblossvals.append(train_return[:-1])

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

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(traj_size / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            # logger.logkv("nupdates", update)
            # logger.logkv("total_timesteps", update * nbatch)
            logger.logkv("fps", fps)
            logger.logkv("completion_ratio", completion_ratio)
            logger.logkv("latent_sample_error", latent_distances)
            logger.logkv("explained_variance", float(ev))
            logger.logkv("episode/reward", safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv("episode/length", safemean([epinfo['l'] for epinfo in epinfobuf]))
            for t in range(task_space.shape[0]):
                logger.logkv("sampled_task/%i" % t, np.sum(tasks[:, t]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv("loss/%s" % lossname, lossval)
            logger.logkv("loss/inference_net", inference_loss)
            logger.dumpkvs()
            if update == 1 and log_folder is not None:
                # save graph to visualize in TensorBoard
                writer = tf.summary.FileWriter(logdir=log_folder, graph=tf.get_default_graph())
                writer.add_graph(tf.get_default_graph())
                writer.flush()
        # if plot_interval and (update % plot_interval == 0 or update == 1):
        #     visualize(model, env, update, plot_folder)
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)
    env.close()
    return model
