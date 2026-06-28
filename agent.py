import numpy as np

class QLearningAgent:

    def __init__(self, n_actions=9, alpha=0.15, gamma=0.97,
            epsilon=1.0, eps_min=0.05, eps_decay=0.994):
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.eps_min   = eps_min
        self.eps_decay = eps_decay
        self.n_states  = 8 * 6 * 5 * 5
        self.Q         = np.ones((self.n_states, self.n_actions)) * 0.5

    def encode_state(self, obs) -> int:
        bins   = [8, 6, 5, 5]
        dims   = [1, 2, 3, 4]
        ranges = [(-1.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        idx    = 0
        stride = 1
        for b, d, (lo, hi) in zip(bins, dims, ranges):
            val  = float(np.clip(obs[d], lo, hi))
            bin_ = int((val - lo) / (hi - lo) * b)
            bin_ = min(bin_, b - 1)
            idx += bin_ * stride
            stride *= b
        return idx

    def select_action(self, obs, greedy=False) -> int:
        state = self.encode_state(obs)
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, obs, action, reward, next_obs, done):
        s      = self.encode_state(obs)
        s_next = self.encode_state(next_obs)
        target = reward + (0.0 if done else self.gamma * np.max(self.Q[s_next]))
        self.Q[s, action] += self.alpha * (target - self.Q[s, action])

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)