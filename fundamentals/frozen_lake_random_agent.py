import gym
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

games = 1000
scores = []
win_pct = []
env = gym.make('FrozenLake-v0')
env.reset()
for i in range(games):
    done = False
    obs = env.reset()
    score = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, done, info  = env.step(action)
        score += reward
    scores.append(score)

    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)

plt.plot(win_pct)
plt.show()