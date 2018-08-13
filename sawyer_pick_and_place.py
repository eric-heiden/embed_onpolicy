from garage.envs.base import Step
from garage.misc.overrides import overrides
from garage.envs.mujoco.sawyer.pick_and_place_env import PickAndPlaceEnv

from dm_control.utils import rewards

import gym
from gym.envs.robotics import rotations
from gym.spaces import Box
from gym.envs.robotics.utils import ctrl_set_action, mocap_set_action, reset_mocap2body_xpos

import numpy as np

# True indicates "pick"-task, False indicates "place"-task
TASKS = [(0.65, 0, 0.15, True)]  # (0.7, -0.3, 0.03, True), (0.7, 0, 0.03, True), (0.7, 0.3, 0.03, True),]
         # (0.7, -0.3, 0.15, False), (0.7, 0, 0.15, False), (0.7, 0.3, 0.15, False)]


class TaskPickAndPlaceEnv(PickAndPlaceEnv):
    def __init__(self, task=0, control_method="position_control", sparse_reward=False, for_her=False, *args, **kwargs):
        self._task = task
        self.onehot = np.zeros(len(TASKS))
        self.onehot[self._task] = 1
        self._step = 0
        self._start_pos = np.array([0.65, 0.0, 0.15])
        self._sparse_reward = sparse_reward
        self.reward_type = "sparse" if sparse_reward else "dense"
        self._max_episode_steps = 30  # used by HER
        self._for_her = for_her

        super().__init__(control_method=control_method, *args, **kwargs)

        self._distance_threshold = 0.03
        reset_mocap2body_xpos(self.sim)

        self.env_setup(self._initial_qpos)

        self.init_qpos = self.sim.data.qpos
        self.init_qvel = self.sim.data.qvel
        self.init_qacc = self.sim.data.qacc
        self.init_ctrl = self.sim.data.ctrl

        self.init_box_height = 0.

        self._gripper_pos = 0.

        self._goal = np.array(TASKS[task][:3])
        self.set_position(self._start_pos)
        print("Instantiating TaskPickAndPlaceEnv (task = %i, control_mode = %s)" % (self._task, self._control_method))

    @overrides
    @property
    def action_space(self):
        if self._control_method == 'torque_control':
            return super(TaskPickAndPlaceEnv, self).action_space
        elif self._control_method == 'position_control':
            # specify lower action limits
            return Box(np.array([-0.01, -0.01, -0.5, 0.]), np.array([0.01, 0.01, 0.5, 1.]), dtype=np.float32)
        else:
            raise NotImplementedError()

    @overrides
    @property
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.get_obs().shape, dtype=np.float32)

    def get_obs(self):
        grip_pos = self.sim.data.get_site_xpos('grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('grip') * dt

        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        object_pos = self.sim.data.get_site_xpos('object0')
        object_rot = rotations.mat2euler(
            self.sim.data.get_site_xmat('object0'))
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velr = self.sim.data.get_site_xvelr('object0') * dt
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp

        achieved_goal = np.squeeze(object_pos.copy())

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
            [int(self._grasp())],
        ])

        if self._for_her:
            return {
                'observation': obs.copy(),
                'achieved_goal': achieved_goal.copy(),
                'desired_goal': self.goal.copy(),
            }
        else:
            if self._control_method == 'torque_control':
                return np.concatenate((self.position, self.sim.data.get_site_xpos('object0') - self.position, [self._gripper_pos, int(self._grasp()), self._step]))
            elif self._control_method == 'position_control':
                return obs  # np.concatenate((self.position, self.sim.data.get_site_xpos('object0') - self.position, [self._gripper_pos, int(self._grasp())]))  # , self._step]))

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.
        d = self._goal_distance(achieved_goal, desired_goal)
        if self._sparse_reward:
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    @overrides
    def step(self, action):
        # no clipping / rescaling of actions
        if self._for_her:
            # actions are in [-1, 1]
            action = (action + 1.) / 2.
            # actions are in [0, 1]
            action = action * (self.action_space.high - self.action_space.low) + self.action_space.low
            action = np.clip(action, self.action_space.low, self.action_space.high)
        # rot_ctrl = np.array([1., 0., 1., 0.])
        # action = np.concatenate([action, rot_ctrl])
        # action, _ = np.split(action, (self.sim.model.nmocap * 7,))
        # action = action.reshape(self.sim.model.nmocap, 7)

        # pos_delta = action[:, :3]
        # action = np.concatenate((np.zeros(2), np.array(action).flatten()))
        action = np.array(action).flatten()
        if self._control_method == "torque_control":
            self.forward_dynamics(action)
            self.sim.forward()
        else:
            reset_mocap2body_xpos(self.sim)
            self.sim.data.mocap_pos[0, :3] = self.sim.data.mocap_pos[0, :3] + action[:3]
            self.set_gripper_state(action[3])
            for _ in range(1):
                self.sim.step()
            self.sim.forward()

        self._step += 1

        # obs = self.get_current_obs()
        # achieved_goal = obs['achieved_goal']
        # goal = obs['desired_goal']


        grasped = self._grasp()  # TODO fix grasp recognition (always returns False)
        object_pos = self.sim.data.get_site_xpos('object0')

        desired_goal = object_pos

        if self._for_her:
            if not grasped:
                achieved_goal = self.position
            else:
                # print("Grasped!")
                achieved_goal = np.squeeze(object_pos.copy())
                desired_goal = self.goal
        else:
            achieved_goal = self.position
        # reward = self._compute_reward(achieved_goal, goal)

        obs = self.get_obs()

        # penalize x/y movement of object
        penalize_2d_motion = np.linalg.norm(object_pos[:2] - np.array(TASKS[self._task][:2]))
        # reward positive change in z direction (up)
        lifting = object_pos[2] - self.init_box_height

        # achieved_dist = self._goal_distance(achieved_goal, self._goal)
        achieved_dist = self._goal_distance(achieved_goal, desired_goal)
        # reward = rewards._sigmoids(self._goal_distance(achieved_goal, goal) / self._goal_distance(self.initial_pos, goal), 0., "cosine")
        # reward = 1. - achieved_dist / self._goal_distance(self._start_pos, self._goal) / 2.  # before
        # reward = 1. - achieved_dist / self._goal_distance(np.zeros(3), self._goal) / 2.
        # TODO sparse reward
        # if grasped:
        #     print("Grasped!")

        # print(self.initial_pos, achieved_goal)

        done = grasped and lifting > 0.005  # (action[3] < 0.2 and not grasped) or grasped and lifting > 0.02  #(achieved_dist < self._distance_threshold)

        if self._for_her:
            reward = self.compute_reward(achieved_goal, desired_goal, {})
        else:
            if self._sparse_reward:
                reward = 0. - penalize_2d_motion + lifting + float(grasped) * 0.3
            else:
                reward = (.3 + lifting * 30.) * float(grasped) + 1. - achieved_dist / self._goal_distance(
                    self._start_pos, object_pos) - penalize_2d_motion

        if done:  # done:
            reward = 20.  # 20.

        if self._for_her:
            info = {
                "l": self._step,
                "r": reward,
                "d": done,
                "grasped": grasped,
                "is_success": grasped and lifting > 0.005
            }
        else:
            info = {
                "episode": {
                    "l": self._step,
                    "r": reward,
                    "d": done,
                    "grasped": grasped,
                    "position": self.position,
                    "task": np.copy(self.onehot)
                }
            }

        # just take gripper position as observation
        return Step(obs, reward, done, **info)

    @overrides
    def reset(self, **kwargs):
        self._step = 0
        super(TaskPickAndPlaceEnv, self).reset()[:3]
        # self.set_position(self._start_pos)
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
        # self.sim.data.set_mocap_pos('mocap', pos[:3])
        # self.sim.data.set_site_pos('object0', pos)

        # self.sim.step()
        # for _ in range(200):
        #     self.sim.step()

        reset_mocap2body_xpos(self.sim)
        self.sim.data.mocap_quat[:] = np.array([0, 1, 0, 0])
        # gripper_ctrl = -50 if gripper_ctrl < 0 else 10
        gripper_ctrl = np.array([0, 0])
        # action = np.concatenate([pos, rot_ctrl, gripper_ctrl])
        # ctrl_set_action(self.sim, action)  # For gripper
        # mocap_set_action(self.sim, action)
        # reset_mocap2body_xpos(self.sim)
        self.sim.data.set_mocap_pos('mocap', pos)
        for _ in range(100):
            self.sim.step()

            reset_mocap2body_xpos(self.sim)
            self.sim.data.mocap_quat[:] = np.array([0, 1, 0, 0])
            self.sim.data.set_mocap_pos('mocap', pos)
            # self.sim.forward()
        # # self.sim.data.mocap_pos[:] = pos
        # reset_mocap2body_xpos(self.sim)
        # print("Set position to", pos)
        # print("SawyerReach Servo Error:", np.linalg.norm(pos-grip_pos))
        self.sim.forward()

    @property
    def start_position(self):
        return self._start_pos.copy()

    def set_start_position(self, point):
        self._start_pos[:] = np.copy(point)

    def select_next_task(self):
        self._task = (self._task + 1) % len(TASKS)
        self.onehot = np.zeros(len(TASKS))
        self.onehot[self._task] = 1
        self._goal = np.array(TASKS[self._task][:3], dtype=np.float32)
        # site_id = self.sim.model.site_name2id('target_pos')
        # self.sim.model.site_pos[site_id] = self._goal
        self.sim.forward()
        return self._task

    def set_gripper_state(self, state):
        # 1 = open, 0 = closed
        state = np.clip(state, 0., 1.)
        # state = np.round(state)
        self._gripper_pos = state
        # self.sim.data.set_joint_qpos('r_gripper_l_finger_joint', state * 0.020833)
        # self.sim.data.set_joint_qpos('r_gripper_r_finger_joint', -state * 0.020833)

        self.sim.data.ctrl[:] = np.array([state * 0.020833, -state * 0.020833])
        for _ in range(3):
            self.sim.step()
        self.sim.forward()
        # new_com = self.sim.data.subtree_com[0]
        # self.dcom = new_com - self.current_com
        # self.current_com = new_com
        # self.sim

    def select_task(self, task: int):
        self._task = task
        self.onehot = np.zeros(len(TASKS))
        self.onehot[self._task] = 1
        self._goal = np.array(TASKS[self._task][:3], dtype=np.float32)

        if TASKS[self._task][3]:
            # pick
            # object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            # r = np.random.randn(3) * 0.2
            # print(r)
            self.set_gripper_state(1)  # open
            for _ in range(20):
                self.sim.step()
            self.set_position(self._start_pos + np.array([0, 0, .1]))  # + np.random.randn(3) * 0.2)
            object_qpos = np.concatenate((TASKS[self._task][:2], [0.03, 0, 1, 0, 1]))
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        else:
            # place
            pos = np.array(self._start_pos[:3]) + np.random.randn(3) * 0.1
            # self._goal = np.array(pos, dtype=np.float32)
            self.set_gripper_state(1)  # open
            # for _ in range(20):
            #     self.sim.step()
            self.set_position(pos)
            # for _ in range(20):
            #     self.sim.step()
            # place object
            offset = np.array([0, 0, -.1])
            object_qpos = np.concatenate((pos + offset, [1, 0, 0, 0]))
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
            # for _ in range(20):
            #     self.sim.step()
            self.set_gripper_state(0)  # closed
            # for _ in range(20):
            #     self.sim.step()

        for _ in range(20):
            self.sim.step()
            # self.sim.forward()

        # site_id = self.sim.model.site_name2id('target_pos')
        # self.sim.model.site_pos[site_id] = self._goal
        self.sim.forward()

        self.init_box_height = self.sim.data.get_site_xpos('object0')[2] + 0.005
        return self._task
