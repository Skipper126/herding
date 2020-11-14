from herding import data
from herding.reward import RewardCounter
import numpy as np
import time


class MediumDistanceRewardCounter(RewardCounter):

    def __init__(self, env_data: data.EnvData):
        self.sheep_count = env_data.config.sheep_count
        self.target_reward = env_data.config.max_episode_reward
        self.herd_target_radius = env_data.config.herd_target_radius
        self.start_distance = 0
        self.previous_distance = 0
        self.total_reward = 0
        self.distance_buffer = env_data.ocl.create_buffer((1,), np.int32)
        self.reward_module = env_data.ocl.create_module(
                                                  'herding/reward/medium_distance/medium_distance.cl',
                                                  'get_medium_distance',
                                                  [env_data.shared_buffers.sheep_positions,
                                                   env_data.shared_buffers.target_position,
                                                   self.distance_buffer])

    def get_reward(self):
        # Corner case when sheep are already in herd at the beginning of episode
        if self.start_distance == 0:
            self.total_reward = self.target_reward  # Trigger is_done flag
            return 0

        medium_distance = self._get_medium_distance()

        reward = (self.previous_distance - medium_distance) * self.target_reward / self.start_distance
        self.previous_distance = medium_distance
        self.total_reward += reward - 0.01
        return reward

    def is_done(self):
        return self.total_reward >= self.target_reward - 0.5  # some space for numeric error

    def reset(self):
        medium_distance = self._get_medium_distance()
        self.start_distance = self.previous_distance = medium_distance
        self.total_reward = 0

    def _get_medium_distance(self):
        self.reward_module.run((self.sheep_count,), (self.sheep_count,))
        medium_distance = self.distance_buffer.map_read()[0]
        self.distance_buffer.unmap()

        return medium_distance

    def get_episode_reward(self):
        return self.total_reward
