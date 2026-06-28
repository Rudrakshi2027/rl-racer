from racing_env import RacingEnv

env = RacingEnv()
obs = env.reset()

print("Starting observation:", obs)
print("Number of actions:", env.n_actions)
print("Number of observations:", env.obs_dim)

# take 10 random steps
import numpy as np
for i in range(10):
    action = np.random.randint(env.n_actions)
    obs, reward, done, info = env.step(action)
    print(f"Step {i+1} | Action: {action} | Reward: {reward:.2f} | Done: {done}")