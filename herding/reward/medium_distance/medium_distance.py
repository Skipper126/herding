from herding import data
from herding.reward import RewardCounter
import numpy as np
import time


class MediumDistanceRewardCounter(RewardCounter):

    def __init__(self, env_data: data.EnvData):
        self.sheep_count = env_data.config.sheep_count
        self.target_reward = env_data.config.max_episode_reward
        self.herd_target_radius = env_data.config.herd_target_radius
        self.time_penalty_rate = env_data.config.time_penalty_rate
        self.start_distance = 0
        self.previous_distance = 0
        self.total_reward = 0
        self.time_penalty_reward_adjustment = 0
        self.out_of_bounds = False
        self.success = False
        self.distance_buffer = env_data.ocl.create_buffer((2,), np.int32)
        self.reward_module = env_data.ocl.create_module(
                                                  'herding/reward/medium_distance/medium_distance.cl',
                                                  'get_medium_distance',
                                                  [env_data.shared_buffers.sheep_positions,
                                                   env_data.shared_buffers.target_position,
                                                   self.distance_buffer,
                                                   env_data.shared_buffers.dogs_positions])

    def get_reward(self):
        # Corner case when sheep are already in herd at the beginning of episode
        if self.start_distance == 0:
            #self.total_reward = self.target_reward  # Trigger is_done flag
            self.success = True
            return 0

        medium_distance, out_of_bounds = self._get_medium_distance()
        reward = (self.previous_distance - medium_distance) * self.target_reward / self.start_distance
        if out_of_bounds == 1:
            self.out_of_bounds = True
            reward -= (100 + self.time_penalty_reward_adjustment * 5)
        if self.total_reward >= self.target_reward - 0.5 - self.time_penalty_reward_adjustment:
            self.success = True
            reward += (100 - self.time_penalty_reward_adjustment * 5)

        self.previous_distance = medium_distance
        self.total_reward += reward
        self._decrease_reward()

        return reward

    def is_done(self):
        return self.out_of_bounds or self.success

    def reset(self):
        medium_distance, _ = self._get_medium_distance()
        self.start_distance = self.previous_distance = medium_distance
        self.total_reward = 0
        self.time_penalty_reward_adjustment = 0
        self.out_of_bounds = self.success = False

    def _get_medium_distance(self):
        self.reward_module.run((self.sheep_count,), (self.sheep_count,))
        medium_distance = self.distance_buffer.map_read()
        self.distance_buffer.unmap()

        return medium_distance[0], medium_distance[1]

    def _decrease_reward(self):
        self.time_penalty_reward_adjustment += self.time_penalty_rate
        self.total_reward -= self.time_penalty_rate

    def get_episode_reward(self):
        return self.total_reward
