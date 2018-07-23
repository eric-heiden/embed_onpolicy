from operator import itemgetter
from typing import Callable

import numpy as np


class BasicCurriculum:
    def __init__(self, env_fn: Callable, unwrap_env: Callable, tasks: int, batches: int):
        self._tasks = tasks
        self._batches = batches
        self._envs = [[env_fn(task=task) for _ in range(batches)] for task in range(tasks)]
        self._unwrap_env = unwrap_env

    @property
    def strategy(self):
        return self._envs

    def update(self, batches):
        pass


class ReverseCurriculum(BasicCurriculum):
    def __init__(self, env_fn: Callable, unwrap_env: Callable, tasks: int, batches: int, delta_steps: int = 2,
                 return_threshold: float = 20, max_progress: int = 20, action_samples: int = 20,
                 distance_threshold: float = 0.05):
        super().__init__(env_fn, unwrap_env, tasks, batches)
        self._starts = [self._unwrap_env(self._envs[task][0]).start_position for task in range(tasks)]
        self._delta_steps = delta_steps
        self._return_threshold = return_threshold
        self._max_progress = max_progress
        self._action_samples = action_samples
        self._distance_threshold = distance_threshold
        self._updates = [[0 for _ in range(batches)] for _ in range(tasks)]
        self._tasks = tasks
        self._batches = batches
        self.initialize()

    def initialize(self):
        for task in range(self._tasks):
            for batch in range(self._batches):
                env = self._envs[task][batch].envs[0]
                env.set_start_position(env.goal)

    def resample(self, task, batch):
        env = self._unwrap_env(self._envs[task][batch])
        start_pos = self._starts[task].copy()
        curr_pos = env.start_position
        samples = []
        for _ in range(self._action_samples):
            env.reset()
            env.set_position(curr_pos)
            for _ in range(self._delta_steps):
                env.step(env.action_space.sample())
            samples.append((np.linalg.norm(start_pos - env.position), env.position))
        samples = sorted(samples, key=itemgetter(0))
        distance, pos = samples[0]
        if distance < self._distance_threshold or self._updates[task][batch] == self._max_progress - 1:
            print("Task %i at batch id %i has reached the start position. (Update %i/%i)" %
                  (task, batch, self._updates[task][batch] + 1, self._max_progress))
            env.set_start_position(start_pos)
            self._updates[task][batch] = self._max_progress
        else:
            print("Updating curriculum for task %i at batch id %i. (Update %i/%i)" %
                  (task, batch, self._updates[task][batch] + 1, self._max_progress))
            env.set_start_position(pos)
        self._updates[task][batch] += 1

    @property
    def strategy(self):
        return self._envs

    def update(self, batches):
        for task in range(self._tasks):
            for batch in range(self._batches):
                if self._updates[task][batch] >= self._max_progress:
                    continue
                (obs, tasks, returns, masks, actions, values, neglogpacs, latents, epinfos,
                 inference_means, inference_stds) = batches[task * self._batches + batch]
                if np.mean(returns) > self._return_threshold or any(epinfo["d"] for epinfo in epinfos):
                    self.resample(task, batch)

    @property
    def progress(self):
        return self._updates

    @property
    def task_progress_ratios(self):
        return [sum(self._updates[task]) / (self._max_progress * self._batches) for task in range(self._tasks)]

    @property
    def max_progress(self):
        return self._max_progress
