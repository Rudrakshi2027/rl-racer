import pygame
import numpy as np
import math
from racing_env import RacingEnv, TRACK_WAYPOINTS, TRACK_WIDTH
from agent import QLearningAgent

WIDTH, HEIGHT = 800, 600
FPS           = 30
WHITE         = (255, 255, 255)
BLACK         = (0,   0,   0)
GRAY          = (50,  50,  50)
GREEN         = (0,   200, 0)
RED           = (200, 0,   0)
YELLOW        = (255, 255, 0)
BLUE          = (0,   150, 255)
SCALE         = 0.9
OFFSET        = np.array([40, 20])

def draw_track(screen):
    wps  = np.array(TRACK_WAYPOINTS)
    half = TRACK_WIDTH / 2
    for i in range(len(wps) - 1):
        seg     = wps[i+1] - wps[i]
        seg_len = np.linalg.norm(seg) + 1e-8
        perp    = np.array([-seg[1], seg[0]]) / seg_len * half
        pts = [wps[i]-perp, wps[i]+perp, wps[i+1]+perp, wps[i+1]-perp]
        pts = [(int(p[0]*SCALE+OFFSET[0]), int(p[1]*SCALE+OFFSET[1])) for p in pts]
        pygame.draw.polygon(screen, GRAY, pts)
    for wp in wps:
        x = int(wp[0]*SCALE+OFFSET[0])
        y = int(wp[1]*SCALE+OFFSET[1])
        pygame.draw.circle(screen, YELLOW, (x, y), 6)

def draw_car(screen, pos, heading):
    x  = int(pos[0]*SCALE+OFFSET[0])
    y  = int(pos[1]*SCALE+OFFSET[1])
    ex = int(x + math.cos(heading)*15)
    ey = int(y + math.sin(heading)*15)
    pygame.draw.circle(screen, BLUE, (x, y), 8)
    pygame.draw.line(screen, WHITE, (x, y), (ex, ey), 2)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RL Racing Agent")
    font   = pygame.font.SysFont("arial", 18)
    env    = RacingEnv()
    agent  = QLearningAgent()
    agent.Q       = np.load("q_table.npy")
    agent.epsilon = 0.0
    obs  = env.reset()
    done = False
    tick = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        if not done:
            action             = agent.select_action(obs, greedy=True)
            obs, _, done, info = env.step(action)
        screen.fill(BLACK)
        draw_track(screen)
        draw_car(screen, env.pos, env.heading)
        text = font.render(
            f"WP: {env.wp_idx}  Speed: {env.speed:.1f}  Steps: {env.step_count}",
            True, WHITE)
        screen.blit(text, (10, 10))
        if done:
            msg = "LAP COMPLETE!" if info["lap_done"] else "CRASHED"
            t   = font.render(msg, True, GREEN if info["lap_done"] else RED)
            screen.blit(t, (WIDTH//2 - 60, HEIGHT//2))
        pygame.display.flip()
        tick.tick(FPS)

main()