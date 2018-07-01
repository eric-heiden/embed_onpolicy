from collections import deque

import numpy as np
import pygame

import gym
import gym.spaces

MAX_SHOWN_TRACES = 10
TRACE_COLORS = [
    (80, 150, 0),
    (100, 180, 10),
    (100, 210, 30),
    (140, 230, 50),
    (180, 250, 150)
]  # yapf: disable
BRIGHT_COLOR = (200, 200, 200)
DARK_COLOR = (150, 150, 150)

TASKS = [(3, 0), (0, 3), (-3, 0), (0, -3)]

MIN_DIST = 0.5

ACTION_LIMIT = 0.3


class PointEnv(gym.Env):
    def __init__(self, show_traces=True):
        self._task = 0
        self._goal = np.array(TASKS[self._task], dtype=np.float32)
        self._point = np.zeros(2)

        self.screen = None
        self.screen_width = 500
        self.screen_height = 500
        self.zoom = 50.
        self.show_traces = show_traces

        self._traces = deque(maxlen=MAX_SHOWN_TRACES)
        self._step = 0

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,))

    @property
    def action_space(self):
        return gym.spaces.Box(low=-ACTION_LIMIT, high=ACTION_LIMIT, shape=(2,))

    @property
    def task(self):
        return self._task

    def select_next_task(self):
        self._task = (self._task + 1) % len(TASKS)
        self._goal = np.array(TASKS[self._task], dtype=np.float32)
        return self._task

    def select_task(self, task: int):
        self._task = task
        self._goal = np.array(TASKS[self._task], dtype=np.float32)
        return self._task

    def reset(self):
        self._point = np.zeros_like(self._goal)
        self._traces.append([tuple(self._point)])
        self._step = 0
        return np.copy(self._point)

    def step(self, action):
        # l, h = -ACTION_LIMIT, ACTION_LIMIT
        # action = action * (h - l) + l
        # action = np.clip(action, -ACTION_LIMIT, ACTION_LIMIT)
        # action *= ACTION_LIMIT
        self._point = self._point + action
        self._traces[-1].append(tuple(self._point))
        self._step += 1

        distance = np.linalg.norm(self._point - self._goal)
        done = distance < MIN_DIST

        reward = 1. - distance / np.linalg.norm(self._goal)

        # completion bonus
        if done and distance < MIN_DIST:
            reward = 20.

        onehot = np.zeros(len(TASKS))
        onehot[self._task] = 1
        info = {
            "episode": {
                "l": self._step,
                "r": reward,
                "task": np.copy(onehot)
            }
        }
        return np.copy(self._point), reward, done, info

    def _to_screen(self, position):
        return (int(self.screen_width / 2 + position[0] * self.zoom),
                int(self.screen_height / 2 - position[1] * self.zoom))

    def render(self, *args, **kwargs):

        if self.screen is None:
            pygame.init()
            caption = "Point Environment"
            pygame.display.set_caption(caption)
            self.screen = pygame.display.set_mode((self.screen_width,
                                                   self.screen_height))

        self.screen.fill((255, 255, 255))

        # draw grid
        for x in range(25):
            dx = -6. + x * 0.5
            pygame.draw.line(self.screen, DARK_COLOR if x % 2 == 0 else BRIGHT_COLOR,
                             self._to_screen((dx, -10)),
                             self._to_screen((dx, 10)))
        for y in range(25):
            dy = -6. + y * 0.5
            pygame.draw.line(self.screen, DARK_COLOR if y % 2 == 0 else BRIGHT_COLOR,
                             self._to_screen((-10, dy)),
                             self._to_screen((10, dy)))

        # draw starting point (blue)
        pygame.draw.circle(self.screen, (0, 0, 255), self._to_screen((0, 0)),
                           10, 0)

        # draw goal (red)
        pygame.draw.circle(self.screen, (255, 40, 0),
                           self._to_screen(self._goal), 10, 0)

        # draw point (green)
        pygame.draw.circle(self.screen, (40, 180, 10),
                           self._to_screen(self._point), 10, 0)

        # draw traces
        if self.show_traces:
            for i, trace in enumerate(self._traces):
                if len(trace) > 1:
                    pygame.draw.lines(
                        self.screen,
                        TRACE_COLORS[-min(len(TRACE_COLORS) - 1, i)],
                        False,
                        [self._to_screen(p) for p in trace])

        pygame.display.flip()

    def terminate(self):
        if self.screen:
            pygame.quit()
