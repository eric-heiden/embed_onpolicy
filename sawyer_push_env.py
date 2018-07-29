import numpy as np

from sawyer_env import SawyerEnv, Configuration


class PushEnv(SawyerEnv):

    def __init__(self, mode: str='horizontal', push_distance=0.3, **kwargs):
        file_path = './garage/vendor/mujoco_models/push.xml'
        self._mode = mode
        self._push_distance = push_distance

        def generate_start_goal():

            # Generate a start position randomly on table
            rand_x = np.random.uniform(low=0.182, high=0.782)
            rand_y = np.random.uniform(low=-0.225, high=0.225)
            start = Configuration(gripper_pos=[rand_x, rand_y, 0.3], gripper_state=1,
                                  object_grasped=False, object_pos=[rand_x, rand_y, 0.03])
            if mode == 'horizontal':
                goal = Configuration(gripper_pos=None, gripper_state=None,
                                     object_grasped=None, object_pos=[rand_x, np.max([-0.225, np.min([0.225, rand_y+self._push_distance])]), 0.03])
            else:
                goal = Configuration(gripper_pos=None, gripper_state=None,
                                     object_grasped=None, object_pos=[np.max([0.182, np.min([0.782, rand_x+self._push_distance])]), rand_y, 0.03])
            return start, goal

        def achieved_goal_fn(env: SawyerEnv):
            return env.object_position

        def desired_goal_fn(env: SawyerEnv):
            return env._goal_configuration.object_pos

        def reward_fn(env: SawyerEnv, achievevd_goal, desired_goal, info: dict):

            d = np.linalg.norm(achievevd_goal - desired_goal, axis=-1)
            if env._reward_type == "sparse":
                return (d < env._distance_threshold).astype(np.float32)

            return -d

        super(PushEnv, self).__init__(
            start_goal_config=generate_start_goal,
            reward_fn=reward_fn,
            file_path=file_path,
            achieved_goal_fn=achieved_goal_fn,
            desired_goal_fn=desired_goal_fn,
            **kwargs)