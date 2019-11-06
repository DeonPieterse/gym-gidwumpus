import gym
import gym_gidwumpus
import random
import numpy as np
import datetime
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm, trange
from time import sleep

env = gym.make('gidwumpus-v0')

actionSpaceSize = env.action_space.n
stateSpaceSize = env.observation_space.n

qTable = np.zeros((stateSpaceSize, actionSpaceSize))

numberOfEpisodes = 10000
maximumStepsPerEpisode = 100

learningRate = 0.1    #learning rate?
discountRate = 0.99    #discount rate?

explorationRate = 2
maxExplorationRate = 1
minExplorationRate = 0.01
explorationDecayRate = 0.001

totalRewardsAllEpisodes = np.zeros(numberOfEpisodes)


def maxAction(qTable, state, actionSpace):
    values = np.array([qTable[state, a] for a in actionSpace])
    maxActionValue = np.argmax(values)
    return actionSpace[maxActionValue]


# Time session starting
timerStart = datetime.datetime.now()

# Progress bar
episodeBar = trange(numberOfEpisodes, desc='Episodes', file=sys.stdout)

for episode in episodeBar:
    if episode % 1000 == 0:
        tqdm.write("starting game %i" % episode)

    state = env.reset()
    done = False
    rewardsCurrentEpisode = 0

    for step in range(maximumStepsPerEpisode):
        rand = np.random.random()

        # Exploration - Exploitation swapping
        explorationRateThreshold = random.uniform(0, 1)
        if explorationRateThreshold > explorationRate:
            #action = np.argmax(qTable[state, :])
            #t, action = np.argmax(qTable[state, :], axis=0)
            action = maxAction(qTable, state, env.unwrapped.actionSpace)
        else:
            action = env.action_space.sample()

        # New action    
        newState, reward, done, info = env.step(action)

        # Update the Q-Table
        x = np.max(qTable[newState, :])
        qTable[state, action] = qTable[state, action] * (1 - learningRate) + learningRate * (reward + discountRate * np.max(qTable[newState, :]))

        # Set the new state
        state = newState

        # Add the new reward
        rewardsCurrentEpisode += reward

        episodeBar.set_postfix(env.render(), refresh=True)

        #episodeBar.set_description(str(episode))
        #env.render()

        # Has the episode ended
        if done:
            break

    # Exploration rate decay
    explorationRate = minExplorationRate + (maxExplorationRate - minExplorationRate) * np.exp(-explorationDecayRate * episode)

    # Add current episode reward to total rewards
    totalRewardsAllEpisodes[episode] = rewardsCurrentEpisode

# Time session ended
timerEnd = datetime.datetime.now()

tqdm.write(str(timerEnd - timerStart))

# Calculate and print the average reward per thousand episodes
count = numberOfEpisodes * 0.1
rewards_per_thousand_episodes = np.split(np.array(totalRewardsAllEpisodes), numberOfEpisodes/count)

tqdm.write("\n\n**********Average reward per thousand episodes**********\n")
for r in rewards_per_thousand_episodes:
    tqdm.write("{0}: {1}".format(count, str(sum(r/1000))))
    count += 1000

# Print updated Q-table
tqdm.write("\n\n**********Q-Table**********\n")
tqdm.write(str(qTable))

plt.plot(totalRewardsAllEpisodes)
plt.show()