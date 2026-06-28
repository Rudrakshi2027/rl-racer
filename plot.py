import numpy as np
import matplotlib.pyplot as plt

rewards    = np.load("rewards.npy")
wp_reached = np.load("wp_reached.npy")
episodes   = np.arange(1, len(rewards) + 1)

def smooth(arr, w=30):
    return np.convolve(arr, np.ones(w)/w, mode='same')

fig, axes = plt.subplots(2, 1, figsize=(10, 6))

axes[0].plot(episodes, rewards, alpha=0.3, color='blue', label='raw')
axes[0].plot(episodes, smooth(rewards), color='blue', linewidth=2, label='smoothed')
axes[0].set_title("Reward per Episode")
axes[0].set_ylabel("Total Reward")
axes[0].legend()

axes[1].plot(episodes, wp_reached, alpha=0.3, color='green', label='raw')
axes[1].plot(episodes, smooth(wp_reached), color='green', linewidth=2, label='smoothed')
axes[1].set_title("Waypoints Reached per Episode")
axes[1].set_ylabel("Waypoints")
axes[1].set_xlabel("Episode")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()
print("Saved training_curves.png")