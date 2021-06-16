import gym
import numpy as np
import matplotlib.pyplot as plt
from q_learning_agent import Agent

#LEFT = 0, DOWN  = 1, RIGHT = 2, UP =3

if __name__ == '__main__':

    games = 500000
    
    scores = []
    win_pct_list = []
 
    env = gym.make('FrozenLake-v0')
    agent = Agent(learning_rate = 0.001, gamma = 0.9, n_actions =4, n_states = 16, eps_start = 1.0, eps_end = 0.01, eps_dec = 0.9999995)
    

    for i in range(games):
        done = False
        obs = env.reset()
        score = 0

        while not done:
            #print(obs)
            action = agent.choose_action(obs)

            #action = env.action_space.sample()
            obs_, reward, done, info  = env.step(action)
            agent.learn(obs , action, reward, obs_)
            score += reward
            obs = obs_
        scores.append(score)

        if i % 100 == 0:
            average = np.mean(scores[-100:])
            win_pct_list.append(average)
            if i % 1000 == 0:
                print('episode ', i, 'win pct %.2f' % average,
                'epsilon %.2f' % agent.epsilon)
    plt.plot(win_pct_list)
    plt.show()