from gym.envs.registration import  register

register(
    id='gidwumpus-v0',
    entry_point='gym_gidwumpus.envs:GidGraphEnv',
)
register(
    id='gidwumpus-extrahard-v0',
    entry_point='gym_gidwumpus.envs:GidWumpusExtraHardEnv',
)