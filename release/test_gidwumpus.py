import gym
import gym_gidwumpus
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

env = gym.make('gidwumpus-v0')


ALPHA = 0.1
GAMMA = 1.0
EPS = 1.0

actionSpaceSize = env.action_space.n
x, y = env.observation_space.shape
stateSpaceSize = x*y

qTable = np.zeros((stateSpaceSize, actionSpaceSize))

numGames = 10000000
totalRewards = np.zeros(numGames)


def maxAction(qTable, state, actions):
    values = np.array([qTable[state, a] for a in actions])
    x, y = action = np.argmax(values, axis=0)
    return actions[y]


for i in range(numGames):
    if i % 5000 == 0:
        print('starting game', i)

        done = False
        episodeRewards = 0
        observation = env.reset()

        while not done:
            rand = np.random.random()
            action = maxAction(qTable, observation, env.unwrapped.actionSpace) if rand < (1-EPS) \
                else env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            episodeRewards += reward
            action = maxAction(qTable, observation, env.unwrapped.actionSpace)
            qTable[observation, action] = qTable[observation, action] \
                + ALPHA * (reward + GAMMA * qTable[observation, action] - qTable[observation, action])
            observation = observation_
            clear_output()
            env.render()

        if EPS - 2 / numGames > 0:
            EPS -= 2 / numGames
        else:
            EPS = 0
        totalRewards[i] = episodeRewards

plt.plot(totalRewards)
plt.show()
