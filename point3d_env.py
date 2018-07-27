from collections import deque

import numpy as np

import gym
import gym.spaces

TASKS = [(3, 0, 0), (-3, 0, 0), (0, 3, 0), (0, -3, 0), (0, 0, 3), (0, 0, -3)]

MIN_DIST = 0.5

ACTION_LIMIT = 0.1  # TODO revert to 0.1


class Point3dEnv(gym.Env):
    def __init__(self, task: int = 0, show_traces=True):
        self._task = task
        self._goal = np.array(TASKS[self._task], dtype=np.float32)
        self._point = np.zeros(3)
        self._start_pos = np.zeros(3)

        self._step = 0

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3, ), dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-ACTION_LIMIT,
            high=ACTION_LIMIT,
            shape=(3, ),
            dtype=np.float32)

    @property
    def task(self):
        return self._task

    @property
    def goal(self):
        return self._goal

    @property
    def position(self):
        return self._point.copy()

    @property
    def start_position(self):
        return self._start_pos.copy()

    def select_next_task(self):
        self._task = (self._task + 1) % len(TASKS)
        self._goal = np.array(TASKS[self._task], dtype=np.float32)
        return self._task

    def select_task(self, task: int):
        self._task = task
        self._goal = np.array(TASKS[self._task], dtype=np.float32)
        return self._task

    def set_start_position(self, point):
        self._start_pos[:] = np.copy(point)

    def set_position(self, point):
        self._point[:] = point

    def reset(self):
        self._point[:] = self._start_pos.copy()
        self._step = 0
        return np.copy(self._point)

    def step(self, action):
        # l, h = -ACTION_LIMIT, ACTION_LIMIT
        # action = action * (h - l) + l
        action = np.clip(action, -ACTION_LIMIT, ACTION_LIMIT)
        # action *= ACTION_LIMIT
        self._point = self._point + action
        self._step += 1

        distance = np.linalg.norm(self._point - self._goal)
        done = distance < np.linalg.norm(self.action_space.low)

        reward = -distance

        # completion bonus
        if done:
            reward = 100.

        onehot = np.zeros(len(TASKS))
        onehot[self._task] = 1
        info = {
            "episode": {
                "l": self._step,
                "r": reward,
                "d": done,
                "task": np.copy(onehot)
            }
        }

        # done = False  # TODO remove

        return np.copy(self._point), reward, done, info

    def render(self, *args, **kwargs):
        raise NotImplementedError()
