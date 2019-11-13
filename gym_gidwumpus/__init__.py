from gym.envs.registration import register

register(
    id='gidwumpus-v1',
    entry_point='gym_gidwumpus.gidwumpus_env:GidWumpusEnv',
)
