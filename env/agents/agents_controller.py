from env import cuda, data
import os


class AgentsController:

    def __init__(self, env_data: data.EnvData):
        self.env_data = env_data

        self.module = cuda.compile_directory(os.getcwd() + '\kernels')
        self.gpu_move_agents = self.module.get_function('move_agents')
        self.gpu_ray_casting = self.module.get_function('ray_casting')

    def move_agents(self, action):
        cuda.memcpy_htod(self.env_data.gpu_action,
                         action)
        self.gpu_move_agents(self.env_data.gpu_dogs_positions,
                             self.env_data.gpu_dogs_rotations,
                             self.env_data.gpu_sheep_positions,
                             self.env_data.gpu_action,
                             block=(1, 1, 1))
        cuda.memcpy_dtoh(self.env_data.dogs_positions,
                         self.env_data.gpu_dogs_positions)
        cuda.memcpy_dtoh(self.env_data.dogs_rotations,
                         self.env_data.gpu_dogs_rotations)
        cuda.memcpy_dtoh(self.env_data.sheep_positions,
                         self.env_data.gpu_sheep_positions)

        return self.env_data.dogs_positions

    def get_observation(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
