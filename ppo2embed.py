import os
import sys
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf

import point_env
from point_env import PointEnv
from policies import MlpEmbedPolicy

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))

from baselines import logger
from collections import deque
from baselines.common import explained_variance
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, task_space, latent_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, task_space, latent_space, nbatch_act, 1, reuse=False)  # type: MlpEmbedPolicy
        train_model = policy(sess, ob_space, ac_space, task_space, latent_space, nbatch_train, nsteps, reuse=True)  # type: MlpEmbedPolicy

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, -CLIPRANGE, CLIPRANGE, name="clip_vf")
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
            print("TRAINABLE VARS", params)
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        # obs, tasks, returns, masks, actions, values, neglogpacs
        def train(lr, cliprange, obs, tasks, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.Observation: obs, train_model.Task: tasks, A: actions, ADV: advs, R: returns, LR: lr,
                      CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs, OLDVPRED: values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]

        def get_latent(task: int):
            one_hot = np.zeros((1, task_space.shape[0]))
            one_hot[0][task] = 1
            latent = sess.run(act_model.Embedding, {act_model.Task: one_hot})
            return latent[0]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

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
        tf.global_variables_initializer().run(session=sess)  # pylint: disable=E1101


class Runner(object):

    def __init__(self, *, env: DummyVecEnv, model: Model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.batch_ob_shape = (nenv * nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.tasks = [e.task for e in env.envs]
        self.onehots = []
        for task in self.tasks:
            one_hot = np.zeros((self.model.task_space.shape[0],))
            one_hot[task] = 1
            self.onehots.append(one_hot)
        self.latents = [model.get_latent(t) for t in self.tasks]
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

        self.lam = lam
        self.gamma = gamma

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_latents, mb_tasks, mb_values, mb_dones, mb_neglogpacs = \
            [], [], [], [], [], [], [], []
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            # self.latents = [self.model.get_latent(t) for t in self.tasks]
            # actions, values, mb_states, neglogpacs = self.model.step(self.latents, self.obs, self.states, self.dones)
            actions, values, mb_states, neglogpacs = self.model.step_from_task(
                self.onehots, self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            actions = np.clip(actions, self.model.action_space.low[0], self.model.action_space.high[0])
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            mb_rewards.append(rewards)

            mb_tasks.append(infos[-1]["episode"]["task"])
            mb_latents.append(self.latents)


            #     done = True
            # else:
            #     mb_tasks.append(infos[-1]["episode"]["task"])
            # for info in infos:
            #     maybeepinfo = info.get('episode')
            #     if maybeepinfo: epinfos.append(maybeepinfo)

            epinfos += [info["episode"] for info in infos]

            if "t" in infos[-1]["episode"] or self.dones[0]:
                # handle malicious sum(rewards) from gym.Monitor
                # print("DONE", rewards)
                self.obs[:] = self.env.reset()
                self.tasks = [e.task for e in self.env.envs]
                self.onehots = []
                for task in self.tasks:
                    one_hot = np.zeros((self.model.task_space.shape[0],))
                    one_hot[task] = 1
                    self.onehots.append(one_hot)
                self.latents = [self.model.get_latent(t) for t in self.tasks]

            # mb_obs.append(self.obs.copy())
            # mb_actions.append(actions)
            # mb_values.append(values)
            # mb_neglogpacs.append(neglogpacs)
            # mb_dones.append(self.dones)
            # mb_latents.append(self.latents)
            # mb_rewards.append(rewards)

            # if self.dones[-1]:
            #     self.obs[:] = self.env.reset()
            #     self.tasks = [e.task for e in self.env.venv.envs]

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_tasks = np.asarray(mb_tasks, dtype=np.float32)
        mb_latents = np.asarray(mb_latents, dtype=np.float32)
        # last_values = self.model.value(self.latents, self.obs, self.states, self.dones)
        last_values = self.model.value_from_task(self.onehots, self.obs, self.states, self.dones)

        print("MEAN reward:", np.mean(mb_rewards))

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0

        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_values

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_tasks, mb_latents, mb_states, epinfos)


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
    colormap = matplotlib.cm.get_cmap("winter")

    fig, axes = plt.subplots(figsize=(ntasks * 4, 8), nrows=2, ncols=ntasks, sharey='row')
    fig.suptitle('Iteration %i' % update)
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

        for sample in range(nsamples):
            obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
            obs[:] = env.reset()
            for pointEnv in env.envs:
                pointEnv._task = t  # TODO expose via setter
                pointEnv._goal = point_env.TASKS[t]
            dones = [False for _ in range(nenv)]
            positions = []

            for _ in range(nsteps):
                # self.latents = [self.model.get_latent(t) for t in self.tasks]
                # actions, values, mb_states, neglogpacs = self.model.step(self.latents, self.obs, self.states, self.dones)
                actions, values, mb_states, neglogpacs = model.step_from_task(
                    onehots, obs, model.initial_state, dones)
                actions = np.clip(actions, model.action_space.low[0], model.action_space.high[0])
                obs[:], rewards, dones, infos = env.step(actions)
                positions.append(np.copy(obs[0]))
                if dones[-1]:
                    break
            positions = np.array(positions)
            ax.scatter(positions[:, 0], positions[:, 1], color=colormap(sample * 1. / nsamples), s=2)
            ax.plot(positions[:, 0], positions[:, 1], color=colormap(sample * 1. / nsamples))

        # visualize latents TODO make actions dependent on these
        ax = axes[1][t]
        ax.grid()
        ticks = np.arange(0, model.latent_space.shape[0], 1)
        ax.set_xticks(ticks)
        for sample in range(nsamples):
            latent = model.get_latent(t)
            ax.scatter(ticks, latent, color=colormap(sample * 1. / nsamples))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(osp.join(plot_folder, 'embed_%05d.png' % update))


def learn(*, policy, env, task_space, latent_space, nsteps, total_timesteps, ent_coef, lr,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
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
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space,
                               task_space=task_space, latent_space=latent_space,
                               nbatch_act=nenvs,
                               nbatch_train=nbatch_train,
                               nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    if load_path is not None:
        model.load(load_path)
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps // nbatch
    for update in range(1, nupdates + 1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, tasks, latents, states, epinfos = runner.run()
        epinfobuf.extend(epinfos)
        mblossvals = []
        if states is None:  # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    # print("mbinds", mbinds)
                    # print("tasks", tasks.shape)
                    slices = (arr[mbinds] for arr in (obs, tasks, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        # else:  # recurrent version
        #     assert nenvs % nminibatches == 0
        #     envsperbatch = nenvs // nminibatches
        #     envinds = np.arange(nenvs)
        #     flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
        #     envsperbatch = nbatch_train // nsteps
        #     for _ in range(noptepochs):
        #         np.random.shuffle(envinds)
        #         for start in range(0, nenvs, envsperbatch):
        #             end = start + envsperbatch
        #             mbenvinds = envinds[start:end]
        #             mbflatinds = flatinds[mbenvinds].ravel()
        #             slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
        #             mbstates = states[mbenvinds]
        #             mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update * nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update * nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
            if update == 1 and log_folder is not None:
                # save graph to visualize in TensorBoard
                writer = tf.summary.FileWriter(logdir=log_folder, graph=tf.get_default_graph())
                writer.add_graph(tf.get_default_graph())
                writer.flush()
        if plot_interval and (update % plot_interval == 0 or update == 1):
            visualize(model, env, update, plot_folder)
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)
    env.close()
    return model
