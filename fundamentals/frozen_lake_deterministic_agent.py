import gym
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#LEFT = 0, DOWN  = 1, RIGHT = 2, UP =3

games = 1000
scores = []
win_pct = []
policy = {0:1, 1:2, 2:1, 3:2,  4:1, 6:2, 8:2, 9:1, 10:1, 11:1, 12:2, 13:2, 14:2}
env = gym.make('FrozenLake-v0')
env.reset()

for i in range(games):
    done = False
    obs = env.reset()
    score = 0

    while not done:
        action = policy[obs]
        #action = env.action_space.sample()
        obs, reward, done, info  = env.step(action)
        score += reward
    scores.append(score)

    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)

plt.plot(win_pct)
plt.show()