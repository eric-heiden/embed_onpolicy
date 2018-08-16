import sys
import os.path as osp
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))
# sys.path.insert(0, '/home/eric/.deep-rl-docker/garage_rjzp_temp')
sys.path.insert(0, osp.join(osp.dirname(__file__), 'garage'))

# from garage.envs.mujoco.sawyer.pusher_env import PusherEnv
from garage.envs.mujoco.sawyer import PushEnv

directions = ("up", "down", "left", "right")
# env = PusherEnv(goal_position=np.array([0.4, 0, 0]), control_method="position_control")
env = PushEnv(control_method="position_control")

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
        action = env.action_space.sample()
        _, r, _, _ = env.step(action)
        if s < 5:
            print(direction, '\t', env.joint_positions)
