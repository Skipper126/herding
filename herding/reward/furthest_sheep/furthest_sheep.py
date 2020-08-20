from herding import data
from herding.reward import RewardCounter
import numpy as np


class FurthestSheepRewardCounter(RewardCounter):

    def __init__(self, env_data: data.EnvData):
        self.sheep_count = env_data.config.sheep_count
        self.target_reward = env_data.config.max_episode_reward
        self.herd_target_radius = env_data.config.herd_target_radius
        self.start_distance = 0
        self.previous_distance = 0
        self.total_reward = 0
        self.distance_buffer = env_data.ocl.create_buffer((2,), np.int32)
        self.reward_module = env_data.ocl.create_module(
                                                  'herding/reward/furthest_sheep/furthest_sheep.cl',
                                                  'get_furthest_sheep_distance',
                                                  [env_data.shared_buffers.sheep_positions,
                                                   env_data.shared_buffers.target_position,
                                                   env_data.shared_buffers.dogs_positions,
                                                   self.distance_buffer])

    def get_reward(self):
        if self.start_distance == 0:  # Corner case when sheep are already in herd at the beginning of episode
            self.total_reward = self.target_reward  # Trigger is_done flag
            return 0

        furthest_sheep_distance, distance_from_dog = self._get_furthest_sheep_distance()

        reward = (self.previous_distance - furthest_sheep_distance) * self.target_reward / self.start_distance
        self.previous_distance = furthest_sheep_distance
        self.total_reward += reward
        return reward

    def is_done(self):
        return self.total_reward >= self.target_reward - 0.5  # some space for numeric error

    def reset(self):
        furthest_sheep_distance, _ = self._get_furthest_sheep_distance()
        self.start_distance = self.previous_distance = furthest_sheep_distance
        self.total_reward = 0

    def _get_furthest_sheep_distance(self):
        self.reward_module.run((self.sheep_count,), (self.sheep_count,))
        furthest_sheep_distance, distance_from_dog = self.distance_buffer.map_read()
        self.distance_buffer.unmap()

        furthest_sheep_distance = max(0, furthest_sheep_distance - self.herd_target_radius)
        return furthest_sheep_distance, distance_from_dog

    def get_episode_reward(self):
        return self.total_reward
