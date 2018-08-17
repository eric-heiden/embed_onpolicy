import sys
import os.path as osp
import numpy as np
from gym.envs.robotics import rotations

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))
# sys.path.insert(0, '/home/eric/.deep-rl-docker/garage_rjzp_temp')
sys.path.insert(0, osp.join(osp.dirname(__file__), 'garage'))

# from garage.envs.mujoco.sawyer.pusher_env import PusherEnv
from garage.envs.mujoco.sawyer import PushEnv, Configuration

directions = ("up", "down", "left", "right")
# env = PusherEnv(goal_position=np.array([0.4, 0, 0]), control_method="position_control")
env = PushEnv(control_method="position_control", easy_gripper_init=True)

# env._start_configuration = Configuration(
#     gripper_pos=np.array([0.55, 0.  , 0.07]),
#     gripper_state=0,
#     object_grasped=False,
#     object_pos=np.array([0.7 , 0.  , 0.03]),
#     joint_pos=np.array([0, -0.96920825,  0.76964638,  2.00488611, -0.56956307, 0.76115281, -0.97169329]))

action = np.zeros(9)
action[7] = -0.03

for i in range(200):
    direction = directions[i % len(directions)]
    env.close()
    # env = PushEnv(direction=direction, easy_gripper_init=True, control_method="position_control")
    env.reset()
    print("Direction:", direction)
    print("Start configuration:", env._start_configuration)
    print("Goal configuration:", env._goal_configuration)
    for s in range(999):
        env.render()
        # action = env.action_space.sample()
        env.step(action)
        # if s < 5:
        #     print(direction, '\t', env.joint_positions)
        grip_rot = rotations.mat2euler(env.sim.data.get_site_xmat('grip'))
        print("gripper orientation:", grip_rot)
