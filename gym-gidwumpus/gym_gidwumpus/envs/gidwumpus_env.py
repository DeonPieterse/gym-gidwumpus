import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SBPB",
        "TEBE",
        "WGPB",
        "TEBP"
    ],
    "8x8": [
        "SEEEEEEE",
        "BEEEBEEE",
        "PBEBPBEE",
        "BBEBBBEB",
        "BPBPBPBP",
        "BPBBBPBB",
        "EBEEEBTE",
        "EEEEETWG",
    ]
}

# TODO: MAP GENERATOR - NOT DONE YET
def generateRandom_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is empty
    """
    valid = False

    # DFS to check that it's a valid path
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0,0))
        while frontier:
            r, c = frontier.pop()
            if not (r,c) in discovered:
                discovered.add((r,c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new > size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if (res[r_new][c_new] not in '#P' or '#W'):
                        frontier.append((r_new, c_new))
        return False
    
    while not valid:
        p = min(1, p)
        res = np.random.choice(['B', 'P', 'T', 'E', 'W'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]

def categoricalSample(probN, npRandom):
    """
    Sample from categorical distribution (What is this?)
    Each row specifies class probabilities
    """
    probN = np.asarray(probN)
    csprobN = np.cumsum(probN) #numpy cumulative sum.
    return (csprobN > npRandom.rand()).argmax()

class GidWumpus(gym.env):
    """
    The Wumpus world is a dangerous one, but for the daring adventurer that dares
    to find the missing pile of gold it is worth the risk.

    S : Start, safe
    B : Breeze, safe
    P : Pit, fall to your doom
    T : Stench, safe but the wumpus is adjacent
    E : Empty, safe
    G : Goal, where the gold is located
    W : Wumpus, this thing will eat you

    The episode ends when you reach the goal, fall in a pit or meet the Wumpus.
    You recieve a reward of 1 if you reach the goal, 0.5 if you reach a stench or a breeze, and zero otherwise
    """
    
    metadata = {'render.modes' : ['human']}

    def __init__(self, nS, nA, P, isd, desc=None, mapName="8x8"):
        self.P = P
        self.isd = isd
        self.lastaction = None # For rendering
        self.nS = nS
        self.nA = nA

        self.actionSpace = spaces.Discrete(self.nA)
        self.observationSpace = spaces.Discrete(self.nS)

        self.seed()
        self.s = categoricalSample(self.isd, self.npRandom)
        self.lastaction = None

        if desc is None and mapName is None:
            # Not done
            #desc = generate_random_map()
            desc = MAPS[mapName] #wrong
        elif desc is None:
            desc = MAPS[mapName]

        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col-1,0)
            elif a == DOWN:
                row = min(row+1,nrow-1)
            elif a == RIGHT:
                col = min(col+1,ncol-1)
            elif a == UP:
                row = max(row-1,0)
            return (row, col)
        
        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        newletter = desc[newrow, newcol]
                        done = bytes(newletter) in b'GH'
                        rew = float(newletter == b'G')
                        li.append((1.0, newstate, rew, done))

    def seed(self, seed=None):
        self.npRandom, seed = seeding.npRandomm(seed)
        return [seed]
    
    def reset(self):
        self.s = categoricalSample(self.isd, self.np_random)
        self.lastaction = None
        return self.s

    def step(self, action):
        transitions = self.P[self.s][a]
        i = categoricalSample([t[0] for t in transitions], self.npRandom)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (s, r, d, {"prob" : p})

        #Returns a list of four things: Next State, Reward, Current State, Bool stating if the current state of model is done and some additional info on our problem.

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        #Gives out information on behaviour of our environment up to present.

    # def close(self):
    #     #incomplete
    #     print()


