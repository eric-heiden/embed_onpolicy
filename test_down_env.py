from sawyer_down_env import DownEnv

env = DownEnv()

for _ in range(200):
    env.reset()
    for _ in range(999):
        env.render()
        action = env.action_space.sample()
        env.step(action)
