The entire environment can be imported as a module into your python
scripts and used as an environment to run with open AI gym's testing framework.

To install the gidwumpus environment, go to gym-gidwumpus root folder and run the command:
pip install -e

This will install the gym environment. We can now use the gym environment with the following code:
""""
import gym
import gym_gidwumpus

env = gym.make('gidwumpus-v0')
""""