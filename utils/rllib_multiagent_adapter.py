from typing import Dict, List
import numpy as np
from ray.rllib import MultiAgentEnv
from herding import Herding


class MultiAgentHerding(MultiAgentEnv):

    def __init__(self, env_config: Dict=None, env: Herding=None):
        self.env = env or Herding(**env_config)
        self.dogs_count = env.env_data.config.dogs_count
        self.agents_names = ['dog_{}'.format(i) for i in range(self.dogs_count)]

    def step(self, action_dict):
        observation, reward, is_done, _ = self.env.step(self._transform_action(action_dict))
        rllib_observation = self._transform_observation(observation)
        rllib_reward = self._transform_reward(reward)
        rllib_dones = self._transform_is_done(is_done)

        return rllib_observation, rllib_reward, rllib_dones, {}

    def reset(self):
        return self._transform_observation(self.env.reset())

    def _transform_action(self, action_dict: Dict[str, np.array]) -> np.array:
        return np.stack(action_dict.values())

    def _transform_observation(self, observation: np.array) -> Dict[str, np.array]:
        dict = {}
        for i, key in enumerate(self.agents_names):
            dict[key] = observation[i]

        return dict

    def _transform_reward(self, reward: int) -> Dict[str, int]:
        dict = {}
        for key in self.agents_names:
            dict[key] = reward

        return dict

    def _transform_is_done(self, is_done: bool) -> Dict[str, bool]:
        dict = {}
        for key in self.agents_names:
            dict[key] = is_done

        dict['__all__'] = is_done

        return dict
