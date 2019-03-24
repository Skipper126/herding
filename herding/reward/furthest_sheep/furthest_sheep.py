from herding import data, opencl
from herding.reward import RewardCounter


class FurthestSheepRewardCounter(RewardCounter):

    def __init__(self, env_data: data.EnvData):
        self.sheep_count = env_data.config.sheep_count
        self.start_distance = 0
        self.previous_distance = 0
        self.total_reward = 0
        self.reward_module = opencl.create_module(env_data,
                                                  'herding/reward/furthest_sheep/furthest_sheep.cl',
                                                  'get_furthest_sheep_distance')
        self.distance_mapping = opencl.create_buffer_mapping(env_data, 'common_output')

    def get_reward(self):
        furthest_sheep_distance = self._get_furthest_sheep_distance()

        reward = (self.previous_distance - furthest_sheep_distance[0]) * 100 / self.start_distance
        self.previous_distance = furthest_sheep_distance
        self.total_reward += reward
        return reward

    def is_done(self):
        pass

    def reset(self):
        furthest_sheep_distance = self._get_furthest_sheep_distance()
        self.start_distance = self.previous_distance = furthest_sheep_distance

    def _get_furthest_sheep_distance(self):
        self.reward_module.run(self.sheep_count)
        furthest_sheep_distance = self.distance_mapping.map_read()[0]
        self.distance_mapping.unmap()

        return furthest_sheep_distance
