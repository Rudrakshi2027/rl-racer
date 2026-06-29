RL_RACER project-Autonomous racing agent 
Name:Rudrakshi Gupta

This project is about building a car that learns to drive on its own. Instead of telling it what to do, we gave it rewards when it did well and penalties when it crashed. Over 900 attempts it learned to navigate a 2D track by itself.For this project we will use 3 python libraries which are numpy, matplotlib and pygame.
The car can't see like humans do. It only gets 5 numbers at every moment:
Speed — how fast is it currently moving
Angle to next checkpoint — is the next checkpoint to my left or right
Distance to next checkpoint — how far away is the next checkpoint
Left wall distance — how close am I to the left wall
Right wall distance — how close am I to the right wall
These 5 numbers are called the state — it's everything the car knows about its situation at any given moment.

Actions — What can the car do?
The car has two controls:
Steer — left, straight, or right
Throttle — accelerate, coast, or brake
Combined that gives 9 possible actions every single step. The agent picks one of these 9 every moment.


Reward Function — How does the car learn what's good and bad?
After every action the car gets a reward number:
Reached a checkpoint → +25 (big reward for making progress)
Moving toward checkpoint → small positive (encourages steady progress)
Moving fast → small positive (encourages the car to actually move)
Near a wall → -2 (discourages crashing into walls)
Crashed off track → -60 (big penalty)
Completed the full lap → +150 (biggest reward possible)
The reward function is the most important design decision because the car learns to do exactly what you reward it for.

Training Results — What happened over 900 episodes?
The car started completely clueless — crashing immediately and reaching zero checkpoints. By episode 900 it was consistently reaching 8-9 out of 10 checkpoints.

Failed Behaviour:We ran an experiment where we changed the reward function to only reward speed — nothing else. No checkpoint bonus, no wall penalty, nothing.The result was the car learned to drive as fast as possible in a straight line and crashed at every single corner. It never learned to navigate the track properly.Why did this happen? Because the agent optimises exactly what you reward it for. If you only reward speed, it learns to go fast — not to turn, not to stay on track, not to reach checkpoints. This proves that how you design your reward function directly controls what behaviour the agent learns.We fixed it by adding checkpoint bonuses and wall penalties back into the reward

Design Choices:
1.Q-learning over DQN — DQN uses a neural network to store the policy. We chose Q-learning because it stores everything in a simple table that you can inspect and understand. For a teaching project, understanding matters more than performance.
2.Discrete actions over continuous — Continuous actions (like exact steering angle) need more complex algorithms like SAC or DDPG. Discrete actions (9 fixed combos) keep things simple and the Q-table manageable.
3.Custom environment over Gymnasium — Gymnasium has ready-made environments like CartPole. But the whole point of this assignment is to design the state, action, and reward yourself. Using Gymnasium's environments would mean we didn't build anything.
4.2D over 3D Trackmania — Trackmania only runs on Windows and needs a GPU for training. Our Mac doesn't have a dedicated GPU. More importantly, the core RL concepts are identical in 2D and 3D — state, action, reward, episode, exploration vs exploitation. 2D is enough to understand all of them.

Files in this project:
1.racing_env.py — The Environment

This file is the world the car lives in. It has 4 main parts:
reset() — Called at the start of every single episode. It puts the car back at checkpoint 0, facing checkpoint 1, with speed 0. Every new attempt starts completely fresh from here.

get_obs() — This is what generates the state. Every step it calculates the 5 numbers the car can see — speed, angle to next checkpoint, distance to next checkpoint, left wall distance, right wall distance. All values are normalised between -1 and 1 so the agent can learn faster.

_wall_distances() — Calculates how far the car is from the left and right walls of the current road segment. It finds the road segment between the current and next waypoint, projects the car's position sideways onto it, and returns both distances. If a distance drops below 2 the car has crashed.

step(action) — The most important method. Takes an action number 0-8, decodes it into steer and throttle, updates the car's heading speed and position using physics, checks if the car reached the next checkpoint, calculates the reward, checks if the episode is over, and returns the new observation reward done and info.

_compute_reward() — Calculates the reward after every action. Gives +25 for reaching a checkpoint, small reward for moving toward the checkpoint, small reward for speed, -2 for being near walls, -60 for crashing, and +150 for completing the lap.




2.agent.py — The Brain
This file contains the Q-learning agent that learns from experience.

__init__() — Sets up all the hyperparameters. Alpha 0.15 is the learning rate — how fast to update Q values. Gamma 0.97 is the discount factor — how much to care about future rewards. Epsilon starts at 1.0 meaning fully random and decays to 0.05 over training. The Q-table is initialised as a 1200 x 9 grid of 0.5 values.

encode_state() — The Q-table needs a single integer as the row index but our observation is 5 continuous floats. This method converts those floats into one integer between 0 and 1199 by dividing each observation into bins and combining them. 8 x 6 x 5 x 5 = 1200 possible states.

select_action() — Epsilon greedy action selection. Generates a random number between 0 and 1. If it is less than epsilon pick a random action — this is exploration. Otherwise pick the action with the highest Q value for the current state — this is exploitation. Early in training epsilon is high so mostly random. Later epsilon is low so mostly learned policy.

update() — The core Q-learning formula. Calculates the target value which is the reward received plus gamma times the maximum Q value of the next state. Then nudges the current Q value toward that target by alpha. Over thousands of steps the Q values converge to accurate estimates of how good each action is in each state.

decay_epsilon() — After every episode multiplies epsilon by 0.994. So epsilon starts at 1.0 and by episode 500 it has dropped to 0.05 where it stays for the rest of training.



3.train.py — The Training Loop
This file connects the environment and agent together and runs 900 episodes.
For each episode it resets the environment, then loops until the episode is done. Inside the loop the agent selects an action, the environment steps with that action, the agent updates its Q-table with what it learned, and the new observation becomes the current one. After each episode epsilon decays and the reward and waypoints are logged. Every 100 episodes it prints the average reward and waypoints so you can see the learning progress. At the end it saves the rewards, waypoints, and Q-table to files.



plot.py — Training Graphs
Loads the saved rewards and waypoints arrays and plots two graphs using matplotlib. The first graph shows total reward per episode — starts negative and rises to around +440. The second graph shows waypoints reached per episode — starts at 0 and rises to 8-9. Both graphs show the raw values and a smoothed line so the trend is clear. Saves the result as training_curves.png.



visualize.py — Pygame Live Demo
Loads the trained Q-table and sets epsilon to 0 so the agent uses its learned policy with no random exploration. Opens a Pygame window showing the track as a grey corridor with yellow waypoint markers. The car is shown as a blue circle with a white arrow showing its heading direction. Every frame the agent picks the best action, the environment steps, and the car's new position is drawn. The top of the screen shows current waypoint, speed, and step count. When the episode ends it shows LAP COMPLETE in green or CRASHED in red.