import numpy as np

from sawyer_env import SawyerEnv, Configuration


class DownEnv(SawyerEnv):
    def __init__(self, **kwargs):
        def generate_start_goal():
            # select a random position within 3 circles
            ind = np.random.randint(0, 3)
            if ind == 0:
                center = self.sim.data.get_geom_xpos('target1')
            elif ind == 1:
                center = self.sim.data.get_geom_xpos('target2')
            else:
                center = self.sim.data.get_geom_xpos('target3')
            # radius = np.random.uniform(0., 1.) * 0.1
            # theta = np.random.uniform(0., 1.) * 2 * np.pi
            #
            # x = radius * np.cos(theta)
            # y = radius * np.sin(theta)
            # pos = np.array([center[0] + x, center[1] + y, 0.025])

            start = Configuration(gripper_pos=np.concatenate([center[:2], [0.35]]), gripper_state=1,
                                  object_grasped=False, object_pos=np.concatenate([center[:2], [0.03]]))
            goal = Configuration(gripper_pos=np.concatenate([center[:2], [0.04]]), gripper_state=1,
                                 object_grasped=False, object_pos=np.concatenate([center[:2], [0.03]]))

            return start, goal

        def reward_fn(env: SawyerEnv, achieved_goal, desired_goal, info: dict):
            d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            if env._reward_type == 'sparse':
                return (d < env._distance_threshold).astype(np.float32)

            # don't deviate in x/y plane
            # if len(info["gripper_position"].shape) == 2:
            #     # vectorized input
            #     d += np.linalg.norm(env._start_configuration.gripper_pos[:2] - info["gripper_position"][:, :2], axis=-1)
            # else:
            #     d += np.linalg.norm(env._start_configuration.gripper_pos[:2] - info["gripper_position"][:2], axis=-1)
            # keep gripper open
            d -= np.mean(info["gripper_state"])
            # don't move the object
            d += np.linalg.norm(env._start_configuration.object_pos - info["object_position"], axis=-1)
            return -d

        super(DownEnv, self).__init__(start_goal_config=generate_start_goal, reward_fn=reward_fn, **kwargs)

    # def compute_reward(self, achieved_goal, desired_goal, info):
    #     # if self._reward_type == 'shaped':
    #         # gripper_pos = self.gripper_position
    #         # object_pos = self.object_position
    #         #
    #         # # penalize x/y movement of object
    #         # penalize_gripper_xy_motion = np.linalg.norm(gripper_pos[:2] - np.array(self._goal[:2]))
    #         # penalize_object_xy_motion = np.linalg.norm(object_pos[:2] - np.array(self._goal[:2]))
    #         #
    #         # achieved_dist = self._goal_distance(gripper_pos, self._goal)
    #         # reward = - achieved_dist / self._goal_distance(self._start_pos, self._goal)
    #         # reward -= penalize_gripper_xy_motion
    #         # reward -= penalize_object_xy_motion
    #     if self._reward_type == 'sparse':
    #         reward = -float(self._goal_distance(achieved_goal, desired_goal) > self._distance_threshold)
    #     else:
    #         reward = -self._goal_distance(achieved_goal, desired_goal)
    #
    #     return reward
