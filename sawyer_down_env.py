import numpy as np

from sawyer_env import SawyerEnv


class DownEnv(SawyerEnv):


    def __init__(self, for_her=True):
        initial_qpos = {
            'right_j0': -0.140923828125,
            'right_j1': -1.2789248046875,
            'right_j2': -3.043166015625,
            'right_j3': -2.139623046875,
            'right_j4': -0.047607421875,
            'right_j5': -0.7052822265625,
            'right_j6': -1.4102060546875,
        }
        if for_her:
            super(DownEnv, self).__init__(
                file_path='./garage/vendor/mujoco_models/pick_and_place.xml',
                initial_goal=np.array([0.65, 0.0, 0.]),
                initial_qpos=initial_qpos,
                start_pos=np.array([0.65, 0.0, 0.1]),
                has_object=True,
                reward_type='dense',
                distance_threshold=0.03
            )
        else:
            super(DownEnv, self).__init__(
                file_path='./garage/vendor/mujoco_models/pick_and_place.xml',
                initial_goal=np.array([0.65, 0.0, 0.]),
                initial_qpos=initial_qpos,
                start_pos=np.array([0.65, 0.0, 0.1]),
                has_object=True,
                reward_type='dense',
                distance_threshold=0.03,
                use_gripper_as_ag=False
            )

    def compute_reward(self, achieved_goal, desired_goal, info):
        if self._reward_type == 'shaped':
            gripper_pos = self.sim.data.get_site_xpos('grip')
            obj_pos = self.sim.data.get_site_xpos('object0')

            # penalize x/y movement of object
            penalize_gripper_xy_motion = np.linalg.norm(gripper_pos[:2] - np.array(self._goal[:2]))
            penalize_object_xy_motion = np.linalg.norm(obj_pos[:2] - np.array(self._goal[:2]))

            achieved_dist = self._goal_distance(gripper_pos, self._goal)
            reward = - achieved_dist / self._goal_distance(self._start_pos, self._goal) - penalize_gripper_xy_motion - penalize_object_xy_motion
        elif self._reward_type == 'sparse':
            reward = -(self._goal_distance(achieved_goal, desired_goal) > self._distance_threshold).astype(np.float32)
        else:
            reward = - self._goal_distance(achieved_goal, desired_goal)

        return reward

    def is_success(self):
        grip_pos = self.sim.data.get_site_xpos('grip')
        object_pos = self.sim.data.get_site_xpos('object0')
        return self._goal_distance(grip_pos, object_pos) < self._distance_threshold

    def is_done(self):
        return self.is_success()