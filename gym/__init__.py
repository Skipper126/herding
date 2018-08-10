from gym.envs.assets.configuration import names
from gym.envs.assets.herding import Herding
from gym.manual_steering import play
import gym


gym.envs.registration.register(
    id='gym-singleDog-v0',
    entry_point='gym.envs:HerdingSingleDog',
    timestep_limit=1000,
    nondeterministic=False
)
