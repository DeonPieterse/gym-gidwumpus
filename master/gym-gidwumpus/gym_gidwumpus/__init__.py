from gym.envs.registration import register

register(
    id='gidwumpus-v0',
    entry_point='gym_gidwumpus.envs.gidwumpus_env:GidWumpusEnv',
)
