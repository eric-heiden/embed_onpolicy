from typing import Callable, Union

import numpy as np

from collections import deque

from model import Model

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv


TimeVarying = Callable[[int], float]


def const_fn(val):
    return lambda _: val


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


class Sampler(object):

    def __init__(self, *, env: DummyVecEnv, unwrap_env: Union[Callable, None], model: Model, gamma, lam,
                 traj_size: int = 20, inference_opt_epochs: int = 4, inference_coef: TimeVarying = const_fn(0.1),
                 use_embedding: bool = True):
        self.env = env
        self.unwrap_env = unwrap_env
        self.model = model
        self.nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        assert (self.nenv == 1)  # ensure to sample from embedding the same number of steps, in training and acting

        self.batch_ob_shape = (self.nenv * traj_size,) + env.observation_space.shape
        self.obs = np.zeros((self.nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.states = model.initial_state
        self.dones = [False for _ in range(self.nenv)]

        self.lam = lam
        self.gamma = gamma
        self.traj_size = traj_size
        self.inference_opt_epochs = inference_opt_epochs
        self.inference_coef = inference_coef
        self.use_embedding = use_embedding

    def run(self, iteration: int, env: DummyVecEnv, task: int, render=None, interactive=False, sample_mean=False):
        mb_obs, mb_rewards, mb_actions, mb_latents, mb_tasks, mb_values, mb_dones, mb_neglogpacs = \
            [], [], [], [], [], [], [], []
        mb_states = self.states

        onehots = []
        latents = []
        one_hot = np.zeros((self.model.task_space.shape[0],))
        one_hot[task] = 1
        for _ in range(self.nenv):
            onehots.append(one_hot)
            latents.append(self.model.get_latent(task))

        # TODO scrap DummyVecEnv
        # assert len(env.envs) == 1
        if self.unwrap_env is None:
            unenv = env
        else:
            unenv = self.unwrap_env(env)

        # for _env in env.envs:
        #     # env.select_task(task)
        #     self.obs[:] = _env.reset()
        self.obs = env.reset()
        # print("Sampling task", env.task, env.start_position, env.position)

        epinfos = []
        completions = 0
        if self.use_embedding:
            traj_window = deque(maxlen=self.model.inference_model.horizon)
        else:
            traj_window = None
        traj_windows = []
        discounts = []

        video = []

        # cumulative stats
        c_obs = []
        c_actions = []
        c_values = []
        c_rewards = []
        c_infos = []

        c_pi_param1 = []
        c_pi_param2 = []

        inference_discounted_log_likelihoods = 0
        inference_means = []
        inference_stds = []

        traj_len = 0

        for step in range(self.traj_size):
            traj_len += 1
            if self.use_embedding:
                pd1, pd2, actions, values, mb_states, neglogpacs = self.model.step(latents, self.obs, onehots, self.states,
                                                                         self.dones,
                                                                         action_type=("mean" if sample_mean else "sample"))
                c_pi_param1.append(pd1[0])
                c_pi_param2.append(pd2[0])
            else:
                pd1, pd2, actions, values, mb_states, neglogpacs = self.model.step(None, self.obs, onehots, self.states,
                                                                         self.dones,
                                                                         action_type=("mean" if sample_mean else "sample"))
                c_pi_param1.append(pd1[0])
                c_pi_param2.append(pd2[0])

            mb_actions.append(actions[0])
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            self.obs[:], rewards, self.dones, infos = env.step(actions)
            self.obs[:] = self.obs[0]
            rewards = rewards[0]
            self.dones = self.dones[0]
            infos = infos[0]

            if render:
                c_obs.append(self.obs.copy())
                c_actions.append(actions)
                c_values.append(values)
                c_rewards.append(rewards)
                c_infos.append(infos)

            if render:
                video.append(render(unenv, c_obs, c_actions, c_values, c_rewards, c_infos))

            if interactive:
                env.render()

            mb_obs.append(self.obs[0].copy())
            mb_rewards.append(rewards)

            mb_tasks.append(one_hot)
            # if not np.array_equal(infos["episode"]["task"], one_hot):
            #     raise Exception("env_task != sample_task", infos["episode"]["task"], one_hot)
            mb_latents.append(np.array(latents).flatten())

            # epinfos += [info["episode"] for info in infos]
            epinfos.append(infos["episode"] if "episode" in infos else {"d": False, "r": 0})

            # if any(self.dones):
            # self.obs[:] = self.env.reset()
            # self.tasks = [e.task for e in self.env.envs]
            # self.onehots = []
            # for task in self.tasks:
            #     one_hot = np.zeros((self.model.task_space.shape[0],))
            #     one_hot[task] = 1
            #     self.onehots.append(np.copy(one_hot))
            # self.latents = [self.model.get_latent(t) for t in self.tasks]

            if step == 0:
                if self.use_embedding:
                    # fill horizon buffer with step 0 copies of trajectory
                    for _ in range(self.model.inference_model.horizon):
                        traj_window.append(np.concatenate((self.obs.flatten(), actions.flatten())))
                discounts.append(self.gamma)
            else:
                discounts.append(discounts[-1] * self.gamma)

            if self.use_embedding:
                traj_window.append(np.concatenate((self.obs.flatten(), actions.flatten())))
                traj_windows.append(np.array(traj_window).flatten())

            # if any(self.dones) and step < self.traj_size-1:
            #     self.obs[:] = self.env.reset()
            #     for _ in range(self.model.inference_model.horizon):
            #         traj_window.append(np.concatenate((self.obs.copy(), actions)))
            #     discounts.append(self.gamma)
            if "episode" in infos and "d" in infos["episode"] and infos["episode"]["d"]:
                completions = 1
                # TODO revert
                # break

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32).flatten()
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32).flatten()
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_tasks = np.asarray(mb_tasks, dtype=np.float32)
        mb_latents = np.asarray(mb_latents, dtype=np.float32)
        last_values = self.model.value(latents, self.obs, onehots, self.states, self.dones)
        traj_windows = np.array(traj_windows)
        discounts = np.array(discounts)

        if self.use_embedding:
            inference_loss, inference_log_likelihoods, inference_discounted_log_likelihoods = 0, [], []
            inference_means, inference_stds = [], []
            # train and evaluate inference network
            for epoch in range(self.inference_opt_epochs):
                idxs = np.arange(traj_len)
                # if epoch < self.inference_opt_epochs - 1:
                #     np.random.shuffle(idxs)
                # TODO shuffle the input for a better training outcome? Is this correct?!
                inference_lll = self.model.inference_model.train(traj_windows[idxs], discounts[idxs], mb_latents[idxs])
                inference_loss, inference_log_likelihood, inference_discounted_log_likelihoods = tuple(inference_lll)
                inference_log_likelihoods.append(inference_log_likelihood)

                inference_params = self.model.inference_model.embedding_params(traj_windows[idxs])
                inference_means = inference_params[0]
                inference_stds = inference_params[1]

        # discount/bootstrap off value fn
        # mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0

        if self.use_embedding:
            # mb_rewards += self.inference_coef * inference_discounted_log_likelihoods.reshape(mb_rewards.shape)
            mb_rewards += self.inference_coef(iteration) * inference_discounted_log_likelihoods

        for t in reversed(range(traj_len)):
            if t == traj_len - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                # if mb_dones[t]:
                #     # XXX reset GAE for this new trajectory piece
                #     lastgaelam = 0
                #     nextnonterminal = 0.
                #     nextvalues = mb_values[t]
                # else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_values

        extras = {}
        if self.model.use_beta:
            extras["alphas"] = np.array(c_pi_param1)
            extras["betas"] = np.array(c_pi_param2)
        else:
            extras["means"] = np.array(c_pi_param1)
            extras["stds"] = np.array(c_pi_param2)
        if self.use_embedding:
            extras["inference_means"] = np.array(inference_means)
            extras["inference_stds"] = np.array(inference_stds)

        if render:
            return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_latents, \
                   mb_tasks, mb_states, epinfos, completions, \
                   inference_discounted_log_likelihoods, inference_means, inference_stds, video, extras

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_latents, \
               mb_tasks, mb_states, epinfos, completions, \
               inference_discounted_log_likelihoods, inference_means, inference_stds, extras
