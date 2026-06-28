import numpy as np
import math

TRACK_WAYPOINTS = [
    (100, 300), (200, 100), (400, 80),  (600, 100),
    (700, 250), (650, 400), (500, 450), (350, 420),
    (200, 450), (100, 380),
]

TRACK_WIDTH = 90
MAX_SPEED   = 6.0
TURN_RATE   = 0.18
DRAG        = 0.92

class RacingEnv:

    def __init__(self):
        self.waypoints = np.array(TRACK_WAYPOINTS, dtype=float)
        self.n_wp      = len(self.waypoints)
        self.n_actions = 9
        self.obs_dim   = 5
        self.reset()

    def reset(self):
        self.pos     = self.waypoints[0].copy()
        self.heading = math.atan2(
            self.waypoints[1][1] - self.waypoints[0][1],
            self.waypoints[1][0] - self.waypoints[0][0])
        self.speed      = 0.0
        self.wp_idx     = 0
        self.done       = False
        self.step_count = 0
        self.trajectory = [self.pos.copy()]
        return self._get_obs()

    def _get_obs(self):
        next_wp   = self.waypoints[(self.wp_idx + 1) % self.n_wp]
        dist_next = np.linalg.norm(next_wp - self.pos)
        dx, dy     = next_wp - self.pos
        target_ang = math.atan2(dy, dx)
        angle_diff = (target_ang - self.heading + math.pi) % (2 * math.pi) - math.pi
        ld, rd = self._wall_distances()
        half   = TRACK_WIDTH / 2
        obs = np.array([
            self.speed / MAX_SPEED,
            angle_diff / math.pi,
            min(dist_next / 400.0, 1.0),
            min(ld / half, 1.0),
            min(rd / half, 1.0),
        ], dtype=np.float32)
        return obs

    def _wall_distances(self):
        wp_curr  = self.waypoints[self.wp_idx % self.n_wp]
        wp_next  = self.waypoints[(self.wp_idx + 1) % self.n_wp]
        seg      = wp_next - wp_curr
        seg_len  = np.linalg.norm(seg) + 1e-8
        seg_unit = seg / seg_len
        perp     = np.array([-seg_unit[1], seg_unit[0]])
        proj     = np.dot(self.pos - wp_curr, perp)
        half     = TRACK_WIDTH / 2
        return max(half - proj, 0), max(half + proj, 0)

    def step(self, action: int):
        steers    = [-1, 0, 1]
        throttles = [-1, 0, 1]
        steer     = steers[action // 3]
        throttle  = throttles[action % 3]
        self.heading += steer * TURN_RATE
        self.speed    = np.clip(self.speed * DRAG + throttle * 1.2,
                                -MAX_SPEED / 2, MAX_SPEED)
        self.pos = self.pos + np.array([
            math.cos(self.heading) * self.speed,
            math.sin(self.heading) * self.speed
        ])
        self.trajectory.append(self.pos.copy())
        self.step_count += 1
        next_wp   = self.waypoints[(self.wp_idx + 1) % self.n_wp]
        dist_next = np.linalg.norm(next_wp - self.pos)
        wp_captured = False
        if dist_next < TRACK_WIDTH / 2:
            self.wp_idx += 1
            wp_captured  = True
        reward = self._compute_reward(wp_captured, dist_next)
        ld, rd    = self._wall_distances()
        off_track = (ld < 2) or (rd < 2)
        lap_done  = self.wp_idx >= self.n_wp - 1
        timeout   = self.step_count >= 1200
        if off_track:
            reward   += -60.0
            self.done = True
        elif lap_done:
            reward   += 150.0
            self.done = True
        elif timeout:
            self.done = True
        info = {
            "wp_idx"   : self.wp_idx,
            "off_track": off_track,
            "lap_done" : lap_done,
            "timeout"  : timeout,
        }
        return self._get_obs(), reward, self.done, info
    def _compute_reward(self, wp_captured, dist_next):
        reward = 0.0
        if wp_captured:
            reward += 25.0
        prev_dist       = getattr(self, '_prev_dist', dist_next)
        progress        = prev_dist - dist_next
        reward         += progress * 0.15
        self._prev_dist = dist_next
        if self.speed > 0:
            reward += self.speed * 0.05
        ld, rd = self._wall_distances()
        half   = TRACK_WIDTH / 2
        if ld < half * 0.3 or rd < half * 0.3:
            reward -= 2.0
        return reward