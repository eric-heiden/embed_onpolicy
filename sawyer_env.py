"""
TODO: fix the sawyer env observation space(should be a dictionary)
"""
import gym
from gym.envs.robotics import rotations
from gym.envs.robotics.utils import reset_mocap_welds, reset_mocap2body_xpos
from gym.spaces import Box

import numpy as np

from garage.envs.mujoco import MujocoEnv
from garage.misc.overrides import overrides

from collections import namedtuple
from typing import Callable
import numpy as np

import gym

Configuration = namedtuple("Configuration", ["gripper_pos", "gripper_state", "object_grasped"])


def default_reward_fn(achieved_goal, desired_goal):
    return -np.linalg.norm(achieved_goal - desired_goal)


class SawyerEnv(gym.GoalEnv):
    def __init__(self,
                 start_configuration: Configuration,
                 goal_configuration: Configuration,
                 reward_fn: Callable = default_reward_fn):
        pass


class SawyerEnv(MujocoEnv, gym.GoalEnv):
    """Sawyer Robot Environments."""

    def __init__(self,
                 file_path,
                 initial_goal=None,
                 initial_qpos=None,
                 start_pos=None,
                 target_range=0.15,
                 reset_above_block=True,
                 distance_threshold=0.05,
                 has_object=False,
                 use_gripper_as_ag=True,
                 max_episode_steps=100,
                 random_obj_start=True,
                 reward_type='sparse',
                 control_method='task_space_control',
                 *args,
                 **kwargs):
        """
        Sawyer Environment.

        :param initial_goal: The initial goal for the goal environment.
        :param initial_qpos: The initial position for each joint.
        :param target_range: delta range the goal is randomized.
        :param args:
        :param kwargs:
        """
        self._initial_goal = initial_goal
        self._initial_qpos = initial_qpos
        self._start_pos = start_pos
        self._target_range = target_range
        self._goal = self._initial_goal
        self._reward_type = reward_type
        self._control_method = control_method
        self._reset_above_block = reset_above_block
        self._distance_threshold = distance_threshold
        self._has_object = has_object
        self._use_gripper_as_ag = use_gripper_as_ag
        self._max_episode_steps = max_episode_steps
        self._step = 0
        self._random_obj_start = random_obj_start
        MujocoEnv.__init__(self, file_path=file_path, *args, **kwargs)
        if initial_qpos is not None:
            self.env_setup(initial_qpos)

    @overrides
    def close(self):
        """Make sure the environment start another viewer next time."""
        if self.viewer is not None:
            self.viewer = None

    def log_diagnostics(self, paths):
        """TODO: Logging."""
        pass

    def env_setup(self, initial_qpos):
        """Set up the robot with initial qpos."""
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        reset_mocap_welds(self.sim)
        self.sim.forward()

    @overrides
    @property
    def action_space(self):
        if self._control_method == 'torque_control':
            return super(SawyerEnv, self).action_space
        elif self._control_method == 'task_space_control':
            # specify lower action limits
            return Box(np.array([-0.01, -0.01, -0.5, 0.]), np.array([0.01, 0.01, 0.5, 1.]), dtype=np.float32)
        else:
            raise NotImplementedError()

    @overrides
    @property
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.get_obs()['observation'].shape, dtype=np.float32)

    def step(self, action):
        action = np.array(action).flatten()
        if self._control_method == "torque_control":
            self.forward_dynamics(action)
            self.sim.forward()
        else:
            action *= 0.05
            reset_mocap2body_xpos(self.sim)
            self.sim.data.mocap_pos[0, :3] = self.sim.data.mocap_pos[0, :3] + action[:3]
            self.sim.data.mocap_quat[0, :4] = np.array([0, 1, 1, 0])
            self.set_gripper_state(action[3])
            for _ in range(1):
                self.sim.step()
            self.sim.forward()

        self._step += 1
        obs = self.get_obs()
        reward_info = dict(obs=obs)
        # if self._has_object:
        #     reward_info = dict(gripper_state=obs.get('gripper_state'), obj_pos=obs.get('obj_pos'))
        # else:
        #     reward_info = dict(gripper_state=obs.get('gripper_state'))

        r = self.compute_reward(
            achieved_goal=obs.get('achieved_goal'),
            desired_goal=obs.get('desired_goal'),
            info=reward_info
        )
        is_success = self.is_success()
        if is_success:
            r = 10
        self.rew += r

        done = self.is_done()
        info = {
            "l": self._step,
            "r": self.rew,
            "d": done,
            "is_success": is_success
        }
        if self._has_object:
            info['grasped'] = obs.get('gripper_state')[1]
        return obs, r, done, info

    def set_gripper_state(self, state):
        # 1 = open, 0 = closed
        state = np.clip(state, 0., 1.)
        self._gripper_pos = state

        self.sim.data.ctrl[:] = np.array([state * 0.020833, -state * 0.020833])
        for _ in range(3):
            self.sim.step()
        self.sim.forward()

    def compute_reward(self, achieved_goal, desired_goal, info):
        raise NotImplementedError

    def get_obs(self):
        grip_pos = self.sim.data.get_site_xpos('grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('grip') * dt

        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        if self._has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            object_rot = rotations.mat2euler(
                self.sim.data.get_site_xmat('object0'))
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
            grasp = [int(self._grasp())]
            obs = np.concatenate([
                grip_pos,
                object_pos.ravel(),  # remove object_pos (reveals task id)
                object_rel_pos.ravel(),
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp,
                qpos,
                qvel,
                grasp,
            ])
            if self._use_gripper_as_ag:
                achieved_goal = np.squeeze(grip_pos.copy())
            else:
                achieved_goal = np.squeeze(object_pos.copy())

            return {
                'observation': obs.copy(),
                'achieved_goal': achieved_goal.copy(),
                'desired_goal': self._goal.copy(),
                'gripper_state': [grip_pos.copy(), grasp[0]],
                'obj_pos': object_pos.copy()
            }

        else:
            achieved_goal = np.squeeze(grip_pos.copy())

            obs = np.concatenate([
                grip_pos,
                grip_velp,
                qpos,
                qvel,
            ])

            return {
                'observation': obs.copy(),
                'achieved_goal': achieved_goal.copy(),
                'desired_goal': self._goal.copy(),
                'gripper_state': [grip_pos.copy()]
            }

    def is_done(self):
        return False

    def is_success(self):
        raise NotImplementedError

    @overrides
    def reset(self, init_state=None):
        self._step = 0
        self.rew = 0
        super(SawyerEnv, self).reset(init_state)

        if self._has_object:
            if self._random_obj_start:
                # select a random position within 3 circles
                ind = np.random.randint(0, 3)
                if ind == 0:
                    center = self.sim.data.get_geom_xpos('target1')
                elif ind == 1:
                    center = self.sim.data.get_geom_xpos('target2')
                else:
                    center = self.sim.data.get_geom_xpos('target3')
                radius = np.random.uniform(0., 1.) * 0.1
                theta = np.random.uniform(0., 1.) * 2 * np.pi

                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                pos = np.array([center[0] + x, center[1] + y, 0.025])
                quat = np.array([1, 0, 0, 0])

                self.sim.data.set_joint_qpos('object0:joint', np.concatenate((pos, quat)))
                self.sim.forward()

        # Move the gripper above the object
        object_pos = self.sim.data.get_site_xpos('object0').copy()
        object_pos[2] += 0.3
        reset_mocap2body_xpos(self.sim)
        self.sim.data.mocap_pos[0, :3] = object_pos
        self.sim.data.mocap_quat[0, :4] = np.array([0, 1, 1, 0])
        self.set_gripper_state(1)
        for _ in range(400):
            self.sim.step()
        self._goal = self._sample_goal()
        return self.get_obs()

    @staticmethod
    def _goal_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def _grasp(self):
        """Determine if the object is grasped"""
        contacts = tuple()
        for coni in range(self.sim.data.ncon):
            con = self.sim.data.contact[coni]
            contacts += ((con.geom1, con.geom2),)
        if ((38, 2) in contacts or (2, 38) in contacts) and ((33, 2) in contacts or (38, 2) in contacts):
            return True
        else:
            return False


def ppo_info(info):
    ppo_infos = {
        "episode": info
    }
    return ppo_infos


class SawyerEnvWrapper():

    def __init__(self, env: SawyerEnv, info_callback=ppo_info, use_max_path_len=True):
        self.env = env
        self._info_callback = info_callback
        self._use_max_path_len = use_max_path_len

    def step(self, action):
        goal_env_obs, r, done, info = self.env.step(action=action)
        if self._use_max_path_len:
            if done or self.env._step >= self.env._max_episode_steps:
                goal_env_obs = self.env.reset()
            else:
                info = dict()
            done = False
        return goal_env_obs.get('observation'), r, done, self._info_callback(info)

    def reset(self, init_state=None):
        goal_env_obs = self.env.reset(init_state)
        return goal_env_obs.get('observation')

    def render(self, mode='human'):
        self.env.render(mode)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space
