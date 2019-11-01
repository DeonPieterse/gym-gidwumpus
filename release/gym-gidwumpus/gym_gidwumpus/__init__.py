from gym.envs.registration import register

register(
    id='gidwumpus-v0',
    entry_point='gym_gidwumpus.gidwumpus_env:GidWumpusEnv',
)
