import itertools
from typing import Dict
import numpy as np
from ray.rllib import MultiAgentEnv
from herding import Herding


class MultiAgentHerding(MultiAgentEnv):

    def __init__(self, env_config: Dict=None, env: Herding=None):
        params = env_config or {}
        self.env = env or Herding(**params)
        self.agents_names = ['dog_{}'.format(i) for i in range(self.env.env_data.config.dogs_count)]

    def step(self, action_dict):
        observation, reward, is_done, _ = self.env.step(np.stack([val for key, val in sorted(action_dict.items())]))
        rllib_observation = dict(zip(self.agents_names, observation))
        rllib_reward = dict(zip(self.agents_names, itertools.cycle([reward])))
        rllib_dones = dict(zip(self.agents_names + ['__all__'], itertools.cycle([is_done])))

        return rllib_observation, rllib_reward, rllib_dones, {}

    def reset(self):
        return dict(zip(self.agents_names, self.env.reset()))
