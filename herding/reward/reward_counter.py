from herding import data
import numpy as np


class RewardCounter:

    def __init__(self, env_data: data.EnvData):
        self.sheep_count = env_data.config.sheep_count
        self.target_reward = env_data.config.max_episode_reward
        self.herd_target_radius = env_data.config.herd_target_radius
        self.start_distance = 0
        self.previous_distance = 0
        self.total_reward = 0
        self.done = False
        self.output_buffer = env_data.ocl.create_buffer((1,), np.int32)
        self.reward_module = env_data.ocl.create_module(
                                                  'herding/reward/reward_counter.cl',
                                                  'get_medium_distance',
                                                  [env_data.shared_buffers.sheep_positions,
                                                   env_data.shared_buffers.target_position,
                                                   self.output_buffer])

    def get_reward(self):
        # Corner case when sheep are already in herd at the beginning of episode
        if self.start_distance == 0:
            self.done = True
            return 0

        medium_distance = self._get_medium_distance()
        reward = (self.previous_distance - medium_distance) * self.target_reward / self.start_distance

        if self.total_reward >= self.target_reward - 0.5:
            reward += 100
            self.done = True

        self.previous_distance = medium_distance
        self.total_reward += reward

        return reward

    def is_done(self):
        return self.done

    def reset(self):
        medium_distance = self._get_medium_distance()
        self.start_distance = self.previous_distance = medium_distance
        self.total_reward = 0
        self.done = False

    def _get_medium_distance(self):
        self.reward_module.run((self.sheep_count,), (self.sheep_count,))
        medium_distance = self.output_buffer.map_read()[0]
        self.output_buffer.unmap()

        return medium_distance

    def get_episode_reward(self):
        return self.total_reward
