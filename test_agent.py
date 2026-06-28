import numpy as np
from racing_env import RacingEnv
from agent import QLearningAgent

env   = RacingEnv()
agent = QLearningAgent()

obs = env.reset()

print("Epsilon:", agent.epsilon)
print("Q-table shape:", agent.Q.shape)
print("State index:", agent.encode_state(obs))

# take 5 steps
for i in range(5):
    action = agent.select_action(obs)
    next_obs, reward, done, info = env.step(action)
    agent.update(obs, action, reward, next_obs, done)
    obs = next_obs
    print(f"Step {i+1} | Action: {action} | Reward: {reward:.2f}")

agent.decay_epsilon()
print("Epsilon after decay:", round(agent.epsilon, 4))