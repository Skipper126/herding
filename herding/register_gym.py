import gym


def register_gym():

    gym.envs.registration.register(
        id='herding-singleDog-v0',
        entry_point='herding:Herding',
        kwargs={
            'dogs_count': 1,
            'sheep_count': 3
        },
    )

    gym.envs.registration.register(
        id='herding-singleDog-v0',
        entry_point='herding:Herding',
        kwargs={
            'dogs_count': 3,
            'sheep_count': 10
        },
    )
