import gym
import numpy as np
from gym import spaces, error, utils
from gym.utils import seeding

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

MAPS = {
    "4x4": [
        ['S', 'B', 'P', 'B'],
        ['T', 'E', 'B', 'E'],
        ['W', 'G', 'P', 'B'],
        ['T', 'E', 'B', 'P']
    ]
}

#ENDSTATES = ['W', 'P', 'G']
ENDSTATES = ['G']

STARTSTATES = ['S']

# REWARDS = {
#     'S': 0,
#     'P': -1,
#     'W': -1,
#     'G': 1,
#     'B': 0.2,
#     'T': 0.2,
#     'E': 0.5
# }

REWARDS = {
    'S': 0,
    'P': -1,
    'W': -1,
    'G': 1,
    'B': 0,
    'T': 0,
    'E': 0
}

class Tile(object):
    def __init__(self, states=[]):
        self.states = states


class Agent(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getXY(self):
        x, y = self.x, self.y
        return (x, y)

    def setXY(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def setX(self, x):
        self.x = x

    def setY(self, Y):
        self.y = Y


class GidWumpusEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, mapName="4x4", nRow=None, nCol=None, hardBorder=True):
        self.hardBorder = hardBorder
        self.grid = MAPS[mapName]
        self.nRow, self.nCol = nRow, nCol = len(self.grid), len(self.grid[0])
        self.stateSpace = np.zeros((nRow, nCol))
        self.actionSpace = [UP, DOWN, LEFT, RIGHT]
        self.endStates = ENDSTATES
        self.startStates = STARTSTATES
        self.rewards = REWARDS
        self.action_space = spaces.Discrete(len(self.actionSpace))
        self.observation_space = spaces.Box(low=0, high=(nRow * nCol), shape=(self.nRow, self.nCol), dtype=np.uint8)
        agentX, agentY = self.index_2d(self.grid, self.startStates[0])
        self.agent = Agent(agentX, agentY)

    def index_2d(self, myList, v):
        for i, x in enumerate(myList):
            if v in x:
                return (i, x.index(v))

    def agentMove(self, currentRow, currentCol, action):
        if self.hardBorder:
            if action == 0:
                currentRow = max(currentRow - 1, 0)
            elif action == 1:
                currentRow = min(currentRow + 1, self.nRow-1)
            elif action == 2:
                currentCol = max(currentCol - 1, 0)
            elif action == 3:
                currentCol = min(currentCol + 1, self.nCol-1)
            return (currentRow, currentCol)
        else:
            if action == 0:
                currentRow = currentRow - 1
            elif action == 1:
                currentRow = currentRow + 1
            elif action == 2:
                currentCol = currentCol - 1
            elif action == 3:
                currentCol = currentCol + 1
            return (currentRow, currentCol)

    def getState(self):
        x, y = self.agent.getXY
        state = self.grid[x][y]
        return state

    def setState(self, state):
        x, y = state
        self.agent.setXY(x, y)

    def getReward(self, state):
        if self.offGridMove(state):
            return -1
        else:
            x, y = state
            return self.rewards.get(self.grid[x][y])

    def offGridMove(self, newState):
        x, y = newState
        if x not in range(len(self.stateSpace)) or y not in range(len(self.stateSpace[0])):
            return True
        else:
            return False

    def isTerminalState(self, state):
        x, y = state
        if self.offGridMove(state):
            return True
        elif self.grid[x][y] in self.endStates:
            return True
        else:
            return False

    def step(self, action):
        x, y = self.agent.getX(), self.agent.getY()
        resultingState = self.agentMove(x, y, action)

        if not self.offGridMove(resultingState):
            self.setState(resultingState)
            reward = self.getReward(resultingState)
            return resultingState, reward, self.isTerminalState((self.agent.getX(), self.agent.getY())), None
        else:
            reward = self.getReward(resultingState)
            self.setState(resultingState)
            return (self.agent.getX(), self.agent.getY()), reward, self.isTerminalState((self.agent.getX(), self.agent.getY())), None

    def reset(self):
        agentX, agentY = self.index_2d(self.grid, self.startStates[0])
        self.agent.setXY(agentX, agentY)
        return (agentX, agentY)

    def render(self, mode='human', close=False):
        for row in self.grid:
            r = ""
            for col in row:
                gridRow, gridCol = (self.grid.index(row), row.index(col))
                x, y = self.agent.getX(), self.agent.getY()
                if (x, y) == (gridRow, gridCol):
                    r = r + "A"
                else:
                    r = r + self.grid[gridRow][gridCol]
            print(r)

    def close(self):
        print()
