import numpy as np
from racing_env import RacingEnv
from agent import QLearningAgent

N_EPISODES = 900

env   = RacingEnv()
agent = QLearningAgent()

rewards    = []
wp_reached = []

print(f"{'Episode':>8}  {'Reward':>10}  {'WP':>4}  {'Epsilon':>8}")
print("-" * 45)

for ep in range(1, N_EPISODES + 1):
    obs  = env.reset()
    done = False
    ep_reward = 0.0

    while not done:
        action                       = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        agent.update(obs, action, reward, next_obs, done)
        obs       = next_obs
        ep_reward += reward

    agent.decay_epsilon()
    rewards.append(ep_reward)
    wp_reached.append(env.wp_idx)

    if ep % 100 == 0 or ep == 1:
        avg_r = np.mean(rewards[-50:])
        avg_w = np.mean(wp_reached[-50:])
        print(f"{ep:>8}  {avg_r:>10.1f}  {avg_w:>4.1f}  {agent.epsilon:>8.4f}")

print("\nTraining complete.")
np.save("rewards.npy",    np.array(rewards))
np.save("wp_reached.npy", np.array(wp_reached))
np.save("q_table.npy",    agent.Q)