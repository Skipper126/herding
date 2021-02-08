import itertools
from typing import Dict

import gym
import numpy as np
from ray.rllib import MultiAgentEnv
from herding import Herding


class HerdingEnvWrapper(MultiAgentEnv):

    def __init__(self, env_config: Dict=None, env: Herding=None):
        params = env_config
        self.env = env or Herding(params)
        self.agents_names = ['dog_{}'.format(i) for i in range(self.env.env_data.config.dogs_count)]

    def step(self, action_dict):
        action = np.stack([[2, val] for key, val in sorted(action_dict.items())])
        observation, reward, is_done, _ = self.env.step(action)
        rllib_observation = dict(zip(self.agents_names, observation))
        rllib_reward = dict(zip(self.agents_names, itertools.cycle([reward])))
        rllib_dones = dict(zip(self.agents_names + ['__all__'], itertools.cycle([is_done])))

        return rllib_observation, rllib_reward, rllib_dones, {}

    def reset(self):
        return dict(zip(self.agents_names, self.env.reset()))

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(3)

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self.env.observation_space
