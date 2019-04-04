from herding import data, opencl
from herding.reward import RewardCounter
import numpy as np


class FurthestSheepRewardCounter(RewardCounter):

    def __init__(self, env_data: data.EnvData):
        self.sheep_count = env_data.config.sheep_count
        self.target_reward = env_data.config.max_episode_reward
        self.start_distance = 0
        self.previous_distance = 0
        self.total_reward = 0
        self.distance_buffer = env_data.ocl.create_buffer((1,), np.int32)
        self.reward_module = env_data.ocl.create_module(
                                                  'herding/reward/furthest_sheep/furthest_sheep.cl',
                                                  'get_furthest_sheep_distance',
                                                  [env_data.shared_buffers.sheep_positions,
                                                   env_data.shared_buffers.target_position,
                                                   self.distance_buffer])

    def get_reward(self):
        furthest_sheep_distance = self._get_furthest_sheep_distance()

        reward = (self.previous_distance - furthest_sheep_distance) * self.target_reward / self.start_distance
        self.previous_distance = furthest_sheep_distance
        self.total_reward += reward
        return reward

    def is_done(self):
        return self.total_reward >= self.target_reward

    def reset(self):
        furthest_sheep_distance = self._get_furthest_sheep_distance()
        self.start_distance = self.previous_distance = furthest_sheep_distance
        self.total_reward = 0

    def _get_furthest_sheep_distance(self):
        self.reward_module.run((self.sheep_count,), (self.sheep_count,))
        furthest_sheep_distance = self.distance_buffer.map_read()[0]
        self.distance_buffer.unmap()

        return furthest_sheep_distance

    def get_episode_reward(self):
        return self.total_reward
