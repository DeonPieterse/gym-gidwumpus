import gym
import gym_gidwumpus
import random
import numpy as np
import datetime
import matplotlib.pyplot as plt
from IPython.display import clear_output

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

for episode in range(numberOfEpisodes):
    if episode % 1000 == 0:
        print('starting game', episode)

    state = env.reset()
    done = False
    rewardsCurrentEpisode = 0

    for step in range(maximumStepsPerEpisode):
        clear_output(wait=True)
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

        env.render()
        
        # Has the episode ended
        if done:
            clear_output(wait=True)
            break

    # Exploration rate decay
    explorationRate = minExplorationRate + (maxExplorationRate - minExplorationRate) * np.exp(-explorationDecayRate * episode)

    # Add current episode reward to total rewards
    totalRewardsAllEpisodes[episode] = rewardsCurrentEpisode

# Time session ended
timerEnd = datetime.datetime.now()

print(timerEnd - timerStart)

# Calculate and print the average reward per thousand episodes
count = 1000
rewards_per_thousand_episodes = np.split(np.array(totalRewardsAllEpisodes), numberOfEpisodes/count)

print("\n\n**********Average reward per thousand episodes**********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

# Print updated Q-table
print("\n\n**********Q-Table**********\n")
print(qTable)

plt.plot(totalRewardsAllEpisodes)
plt.show()