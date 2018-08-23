import gym


gym.envs.registration.register(
    id='herding-singleDog-v0',
    entry_point='herding.openaigym.envs:HerdingSingleDog',
    timestep_limit=1000,
    nondeterministic=False
)
