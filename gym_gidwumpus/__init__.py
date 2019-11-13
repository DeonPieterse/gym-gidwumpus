from gym.envs.registration import register

register(
    id='gidwumpus-v1',
    entry_point='envs.gidwumpus_env:GidWumpusEnv',
)
