from gym.envs.registration import register

register(
    id='BipedalWalkerHardcoreStump1-v0',
    entry_point='custom_env.bipedal_walker_hardcore_stump1:BipedalWalkerHardcoreEdit2'
)

register(
    id='BipedalWalkerHardcoreStateout-v2',
    entry_point='custom_env.bipedal_walker_hardcore-v2-stateout:BipedalWalkerHardcoreEdit2'
)