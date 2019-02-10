from herding import data, cuda
from herding.reward import RewardCounter
import os
import numpy as np


class FurthestSheepRewardCounter(RewardCounter):

    def __init__(self, env_data: data.EnvData):
        self.device_buffer = env_data.device_buffer
        self.sheep_count = env_data.config.sheep_count
        self.furthest_sheep_distance = np.ndarray((1,), dtype=np.int32)
        self.start_distance = 0
        self.previous_distance = 0
        self.total_reward = 0
        kernels_dir = os.path.join(os.path.dirname(__file__), 'kernels')
        self.module = cuda.compile_files([os.path.join(kernels_dir, 'furthest_sheep.cu')],
                                         header_files=[data.get_env_data_header_path()],
                                         template=env_data.config._asdict())
        self.get_furthest_sheep_distance = self.module.get_function('get_furthest_sheep_distance')
        self.device_furthest_sheep_distance = self.module.get_global('distance')[0]

    def get_reward(self):
        self.get_furthest_sheep_distance(self.device_buffer, block=(self.sheep_count, 1, 1))
        cuda.memcpy_dtoh(self.furthest_sheep_distance, self.device_furthest_sheep_distance)
        reward = (self.previous_distance - self.furthest_sheep_distance[0]) *100 / self.start_distance
        self.previous_distance = self.furthest_sheep_distance[0]
        self.total_reward += reward
        return reward

    def is_done(self):
        pass

    def reset(self):
        self.get_furthest_sheep_distance(self.device_buffer, block=(self.sheep_count, 1, 1))
        cuda.memcpy_dtoh(self.furthest_sheep_distance, self.device_furthest_sheep_distance)
        self.start_distance = self.previous_distance = self.furthest_sheep_distance[0]
