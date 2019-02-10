import gym

def register_gym():

    gym.envs.registration.register(
        id='herding-singleDog-v0',
        entry_point='herding:Herding',
        kwargs={
            'dog_count': 3
        },
        timestep_limit=1000,
        nondeterministic=False
    )
