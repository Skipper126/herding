from herding.envs.assets import constants
from herding.envs.assets.herding import Herding
from herding.manual_steering import play

import gym


class Envs:
    HERDING_SINGLE_DOG ='herding-singleDog-v0'


gym.envs.registration.register(
    id=Envs.HERDING_SINGLE_DOG,
    entry_point='herding.envs:HerdingSingleDog',
    timestep_limit=1000,
    nondeterministic=False
)

