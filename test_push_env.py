import sys
import os.path as osp
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))
sys.path.insert(0, '/home/eric/garage_pushenv')
# sys.path.insert(0, osp.join(osp.dirname(__file__), 'garage'))

from garage.envs.mujoco.sawyer import PushEnv, PickAndPlaceEnv

directions = ("up", "down", "left", "right")
env = PushEnv(direction=directions[0], control_method="task_space_control")

for i in range(200):
    direction = directions[i % len(directions)]
    env.close()
    env = PushEnv(direction=direction, control_method="task_space_control")
    env.reset()
    print("Direction:", direction)
    print("Start configuration:", env._start_configuration)
    print("Goal configuration:", env._goal_configuration)
    for _ in range(999):
        env.render()
        action = np.array([0.1, 0, 0])  # np.zeros_like(env.action_space.sample())
        _, r, _, _ = env.step(action)
        print(r)
