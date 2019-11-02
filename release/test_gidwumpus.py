import gym
import gym_gidwumpus
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import datetime

env = gym.make('gidwumpus-v0')

#env.unwrapped.hardBorder = True

ALPHA = 0.1
GAMMA = 1.0
EPS = 1.0

actionSpaceSize = env.action_space.n
x, y = env.observation_space.shape
stateSpaceSize = x*y

qTable = np.zeros((stateSpaceSize, actionSpaceSize))

numGames = 50000
#numGames = 100000000

totalRewards = np.zeros(numGames)


def maxAction(qTable, state, actions):
    values = np.array([qTable[state, a] for a in actions])
    x, y = np.argmax(values, axis=0)
    return actions[y]


timerStart = datetime.datetime.now()
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
        # print('======')
        # env.render()
        # print('======')

    if EPS - 2 / numGames > 0:
        EPS -= 2 / numGames
    else:
        EPS = 0
    totalRewards[i] = episodeRewards
timerEnd = datetime.datetime.now()

print(timerEnd - timerStart)

print(qTable)
plt.plot(totalRewards)
plt.show()