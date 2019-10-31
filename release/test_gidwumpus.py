import gym
import gym_gidwumpus

env = gym.make('gidwumpus-v0')

for episode in range(100):
    observation = env.reset()
    print('Observation RESETED', observation)
    for t in range(100):
        print('Observation before action: ', observation)
        action = env.action_space.sample()
        print('Action: ', action)
        observation, reward, done, info = env.step(action)
        print('Observation after action: ', observation)
        print('Reward: ', reward)
        print('Done: ', done)
        print('Info: ', info)
        env.render()
        if reward == 1:
            print("---------------------------------------------------------------------------------------------------------------------------------------")
        print("||")
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("Episode ended: {}\n".format(episode+1))
            print("========================================================================================================================================\n")
            break
