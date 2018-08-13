import sys
import os.path as osp
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))
sys.path.insert(0, osp.join(osp.dirname(__file__), 'garage'))

from garage.envs.mujoco.sawyer import PushEnv

directions = ("up", "down", "left", "right")
env = PushEnv(direction=directions[0], control_method="position_control")

for i in range(200):
    direction = directions[i % len(directions)]
    env.close()
    env = PushEnv(direction=direction, easy_gripper_init=True, control_method="position_control")
    env.reset()
    print("Direction:", direction)
    print("Start configuration:", env._start_configuration)
    print("Goal configuration:", env._goal_configuration)
    for s in range(999):
        env.render()
        action = np.zeros_like(env.action_space.sample())
        _, r, _, _ = env.step(action)
        if s < 5:
            print(direction, '\t', env.joint_positions)
