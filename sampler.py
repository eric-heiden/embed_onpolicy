import numpy as np

from collections import deque

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

import point_env
from model import Model


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


class Sampler(object):

    def __init__(self, *, env: DummyVecEnv, model: Model, gamma, lam, traj_size: int = 20, inference_opt_epochs: int = 4, inference_coef: float = 0.1):
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
        self.inference_coef = inference_coef

    def run(self, task: int):
        mb_obs, mb_rewards, mb_actions, mb_latents, mb_tasks, mb_values, mb_dones, mb_neglogpacs = \
            [], [], [], [], [], [], [], []
        mb_states = self.states

        onehots = []
        latents = []
        for _ in range(self.env.num_envs):
            one_hot = np.zeros((self.model.task_space.shape[0],))
            one_hot[task] = 1
            onehots.append(one_hot)
            latents.append(self.model.get_latent(task))

        for env in self.env.envs:  # type: point_env.PointEnv
            env.select_task(task)
            self.obs[:] = env.reset()

        epinfos = []
        completions = 0
        traj_window = deque(maxlen=self.model.inference_model.horizon)
        traj_windows = []
        discounts = []

        for step in range(self.traj_size):
            actions, values, mb_states, neglogpacs = self.model.step(latents, self.obs, onehots, self.states, self.dones)
            # actions, values, mb_states, neglogpacs = self.model.step_from_task(
            #     self.onehots, self.obs, self.states, self.dones)
            # actions = np.clip(actions, self.model.action_space.low[0], self.model.action_space.high[0])
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            mb_obs.append(self.obs.copy())
            mb_rewards.append(rewards)

            mb_tasks.append(infos[-1]["episode"]["task"])
            mb_latents.append(np.array(latents).flatten())

            epinfos += [info["episode"] for info in infos]

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
                # fill horizon buffer with step 0 copies of trajectory
                for _ in range(self.model.inference_model.horizon):
                    traj_window.append(np.concatenate((self.obs.copy(), actions)))
                discounts.append(self.gamma)
            else:
                discounts.append(discounts[-1] * self.gamma)

            traj_window.append(np.concatenate((self.obs.copy(), actions)))
            traj_windows.append(np.array(traj_window).flatten())

            # if any(self.dones) and step < self.traj_size-1:
            #     self.obs[:] = self.env.reset()
            #     for _ in range(self.model.inference_model.horizon):
            #         traj_window.append(np.concatenate((self.obs.copy(), actions)))
            #     discounts.append(self.gamma)
            if any([info["episode"]["d"] for info in infos]):
                completions = 1

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_tasks = np.asarray(mb_tasks, dtype=np.float32)
        mb_latents = np.asarray(mb_latents, dtype=np.float32)
        last_values = self.model.value(latents, self.obs, onehots, self.states, self.dones)
        traj_windows = np.array(traj_windows)
        discounts = np.array(discounts)

        inference_loss, inference_log_likelihoods, inference_discounted_log_likelihoods = 0, [], []
        inference_means, inference_stds = [], []
        # train and evaluate inference network
        for epoch in range(self.inference_opt_epochs):
            idxs = np.arange(self.traj_size)
            # if epoch < self.inference_opt_epochs - 1:
            #     np.random.shuffle(idxs)
            # TODO shuffle the input for a better training outcome? Is this correct?!
            inference_lll = self.model.inference_model.train(traj_windows[idxs], discounts[idxs], mb_latents[idxs])
            inference_loss, inference_log_likelihood, inference_discounted_log_likelihoods = tuple(inference_lll)
            inference_log_likelihoods.append(inference_log_likelihood)

            inference_params = self.model.inference_model.embedding_params(traj_windows[idxs])
            inference_means += list(inference_params[0])
            inference_stds += list(inference_params[1])

        # discount/bootstrap off value fn
        # mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0

        mb_rewards += self.inference_coef * inference_discounted_log_likelihoods.reshape(mb_rewards.shape)  # TODO use discounted LL?

        for t in reversed(range(self.traj_size)):
            if t == self.traj_size - 1:
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

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)), mb_latents,
                mb_tasks, mb_states, epinfos, completions, inference_loss, inference_log_likelihoods, inference_discounted_log_likelihoods,
                inference_means, inference_stds)