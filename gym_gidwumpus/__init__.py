from gym.envs.registration import register

register(
    id='gidwumpus-v1',
    entry_point='gidwumpus_env:GidWumpusEnv',
)
