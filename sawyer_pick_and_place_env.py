import numpy as np

from garage.misc.overrides import overrides
from gym.spaces import Box

from sawyer_env import SawyerEnv, Configuration


class PickAndPlaceEnvEnv(SawyerEnv):
    def __init__(self, **kwargs):
        def start_goal_config():
            center = self.sim.data.get_geom_xpos('target2')
            start = Configuration(gripper_pos=np.concatenate([center[:2], [0.35]]), gripper_state=1,
                                  object_grasped=False, object_pos=np.concatenate([center[:2], [0.03]]))
            goal = Configuration(gripper_pos=np.concatenate([center[:2], [0.105]]), gripper_state=0,
                                 object_grasped=True, object_pos=np.concatenate([center[:2], [0.1]]))
            return start, goal

        def reward_fn(env: SawyerEnv, achieved_goal, desired_goal, info: dict):
            d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            if env._reward_type == 'sparse':
                return (d < env._distance_threshold).astype(np.float32)

            return - d

        def success_fn(env: SawyerEnv, _achieved_goal, _desired_goal, _info: dict):
            return env.has_object and env.object_position[2] >= self._goal_configuration.object_pos[2]

        super(PickAndPlaceEnvEnv, self).__init__(start_goal_config=start_goal_config,
                                       reward_fn=reward_fn, success_fn=success_fn, **kwargs)

    def get_obs(self):
        gripper_pos = self.gripper_position
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('grip') * dt

        object_pos = self.object_position
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velp -= grip_velp
        grasped = self.has_object
        obs = np.concatenate([
            gripper_pos,
            object_pos
        ])

        achieved_goal = self._achieved_goal_fn(self)
        desired_goal = self._desired_goal_fn(self)

        achieved_goal_qpos = np.concatenate((achieved_goal, [1, 0, 0, 0]))
        self.sim.data.set_joint_qpos('achieved_goal:joint', achieved_goal_qpos)
        desired_goal_qpos = np.concatenate((desired_goal, [1, 0, 0, 0]))
        self.sim.data.set_joint_qpos('desired_goal:joint', desired_goal_qpos)

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
            'gripper_state': self.gripper_state,
            'gripper_pos': gripper_pos.copy(),
            'has_object': grasped,
            'object_pos': object_pos.copy()
        }

    @overrides
    @property
    def action_space(self):
        return Box(np.array([-0.1, -0.1, -0.1, -1.]), np.array([0.1, 0.1, 0.1, 1.]), dtype=np.float32)
