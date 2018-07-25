import sys
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(__file__), 'baselines'))
sys.path.insert(0, osp.join(osp.dirname(__file__), 'garage'))

from sawyer_down_env import DownEnv

env = DownEnv()

for _ in range(200):
    env.reset()
    for _ in range(999):
        env.render()
        action = env.action_space.sample()
        env.step(action)
