from garage.envs.base import Step
from garage.misc.overrides import overrides
from garage.envs.mujoco.sawyer.reacher_env import ReacherEnv

from dm_control.utils import rewards

import gym
from gym.spaces import Box
from gym.envs.robotics.utils import mocap_set_action, reset_mocap2body_xpos

import numpy as np


TASKS = [
    # (0.6, 0., 0.60),
    (0.3, -0.3, 0.30), (0.3, 0.3, 0.30),
    (0.6, 0.0, 0.8)]


class TaskReacherEnv(ReacherEnv):
    def __init__(self, task=0, control_method="position_control", *args, **kwargs):
        self._task = task
        self.onehot = np.zeros(len(TASKS))
        self.onehot[self._task] = 1
        self._step = 0
        self._start_pos = np.array([0.8, 0.0, 0.15])
        super().__init__(control_method=control_method, *args, **kwargs)
        self._distance_threshold = 0.03
        reset_mocap2body_xpos(self.sim)

        self.init_qpos = self.sim.data.qpos
        self.init_qvel = self.sim.data.qvel
        self.init_qacc = self.sim.data.qacc
        self.init_ctrl = self.sim.data.ctrl

        self._goal = np.array(TASKS[task])
        self.set_position(self._start_pos)
        print("Instantiating TaskReacherEnv (task = %i, control_mode = %s)" % (self._task, self._control_method))

    @overrides
    @property
    def action_space(self):
        if self._control_method == 'torque_control':
            return super(TaskReacherEnv, self).action_space
        elif self._control_method == 'position_control':
            # specify lower action limits
            return Box(-0.03, 0.03, shape=(3,), dtype=np.float32)
        else:
            raise NotImplementedError()

    @overrides
    @property
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.get_obs().shape, dtype=np.float32)

    def get_obs(self):
        if self._control_method == 'torque_control':
            return np.concatenate((self.position, self.sim.data.qpos))
        elif self._control_method == 'position_control':
            return self.position

    @overrides
    def step(self, action):
        # no clipping / rescaling of actions
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        # rot_ctrl = np.array([1., 0., 1., 0.])
        # action = np.concatenate([action, rot_ctrl])
        # action, _ = np.split(action, (self.sim.model.nmocap * 7,))
        # action = action.reshape(self.sim.model.nmocap, 7)

        # pos_delta = action[:, :3]
        if self._control_method == "torque_control":
            self.forward_dynamics(action)
            self.sim.forward()
        else:
            reset_mocap2body_xpos(self.sim)
            self.sim.data.mocap_pos[:] = self.sim.data.mocap_pos + action
            for _ in range(50):
                self.sim.step()
            self._step += 1

        # obs = self.get_current_obs()
        # achieved_goal = obs['achieved_goal']
        # goal = obs['desired_goal']
        achieved_goal = self.position
        # reward = self._compute_reward(achieved_goal, goal)

        obs = self.get_obs()

        achieved_dist = self._goal_distance(achieved_goal, self._goal)
        # reward = rewards._sigmoids(self._goal_distance(achieved_goal, goal) / self._goal_distance(self.initial_pos, goal), 0., "cosine")
        # reward = 1. - achieved_dist / self._goal_distance(self._start_pos, self._goal) / 2.  # before
        # reward = 1. - achieved_dist / self._goal_distance(np.zeros(3), self._goal) / 2.
        # TODO sparse reward
        reward = 1. - achieved_dist / self._goal_distance(self._start_pos, self._goal)

        # print(self.initial_pos, achieved_goal)

        done = (achieved_dist < self._distance_threshold)

        if done:
            reward = 20.  # 20.

        info = {
            "episode": {
                "l": self._step,
                "r": reward,
                "d": done,
                "position": self.position,
                "task": np.copy(self.onehot)
            }
        }
        # just take gripper position as observation
        return Step(obs, reward, False, **info)

    @overrides
    def reset(self, **kwargs):
        self._step = 0
        super(TaskReacherEnv, self).reset()[:3]
        self.set_position(self._start_pos)
        self.select_task(self._task)
        return self.get_obs()

    @property
    def task(self):
        return self._task

    @property
    def goal(self):
        return self._goal

    @property
    def position(self):
        return self.sim.data.get_site_xpos('grip')

    def set_position(self, pos):
        self.sim.data.set_mocap_pos('mocap', pos)

        self.sim.step()
        for _ in range(200):
            self.sim.step()

        # grip_pos = self.sim.data.get_site_xpos('grip')
        # reset_mocap2body_xpos(self.sim)
        # self.sim.data.set_mocap_pos('mocap', pos)
        # # self.sim.data.mocap_pos[:] = pos
        # reset_mocap2body_xpos(self.sim)
        # print("Set position to", pos)
        # print("SawyerReach Servo Error:", np.linalg.norm(pos-grip_pos))
        # self.sim.forward()

    @property
    def start_position(self):
        return self._start_pos.copy()

    def set_start_position(self, point):
        self._start_pos[:] = np.copy(point)

    def select_next_task(self):
        self._task = (self._task + 1) % len(TASKS)
        self.onehot = np.zeros(len(TASKS))
        self.onehot[self._task] = 1
        self._goal = np.array(TASKS[self._task], dtype=np.float32)
        site_id = self.sim.model.site_name2id('target_pos')
        self.sim.model.site_pos[site_id] = self._goal
        self.sim.forward()
        return self._task

    def select_task(self, task: int):
        self._task = task
        self.onehot = np.zeros(len(TASKS))
        self.onehot[self._task] = 1
        self._goal = np.array(TASKS[self._task], dtype=np.float32)
        site_id = self.sim.model.site_name2id('target_pos')
        self.sim.model.site_pos[site_id] = self._goal
        self.sim.forward()
        return self._task
