import gym
import gym_gidwumpus
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import datetime

env = gym.make('gidwumpus-v0')

#env.unwrapped.hardBorder = True

ALPHA = 0.01
GAMMA = 1.0
EPS = 1.0

actionSpaceSize = env.action_space.n
x, y = env.observation_space.shape
stateSpaceSize = x*y

qTable = np.zeros((stateSpaceSize, actionSpaceSize))

numberOfEpisodes = 1000
#numberOfEpisodes = 100000000

maximumStepsPerEpisode = 100

totalRewards = np.zeros(numberOfEpisodes)


def maxAction(qTable, state, actions):
    values = np.array([qTable[state, a] for a in actions])
    x, y = np.argmax(values, axis=0)
    return actions[y]


timerStart = datetime.datetime.now()
for i in range(numberOfEpisodes):
    if i % 1000 == 0:
        print('starting game', i)

    state = env.reset()
    done = False
    episodeRewards = 0

    for step in range(maximumStepsPerEpisode):
        if not done:
            rand = np.random.random()
            action = maxAction(qTable, state, env.unwrapped.actionSpace) if rand < (1-EPS) \
                else env.action_space.sample()
            newState, reward, done, info = env.step(action)
            episodeRewards += reward
            action = maxAction(qTable, state, env.unwrapped.actionSpace)
            qTable[state, action] = qTable[state, action] \
                + ALPHA * (reward + GAMMA * qTable[state, action] - qTable[state, action])
            state = newState
            # print('======')
            # env.render()
            # print('======')
    if EPS - 2 / numberOfEpisodes > 0:
        EPS -= 2 / numberOfEpisodes
    else:
        EPS = 0

    totalRewards[i] = episodeRewards
timerEnd = datetime.datetime.now()

print(timerEnd - timerStart)

plt.plot(totalRewards)
plt.show()