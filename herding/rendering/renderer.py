from gym.envs.classic_control import rendering
from herding.rendering.geoms import dog_geom, sheep_geom, target_geom
from herding import data, opencl
from typing import NamedTuple
import numpy as np


class Arrays(NamedTuple):
    dogs_positions: np.ndarray
    sheep_positions: np.ndarray
    target_position: np.ndarray
    observation: np.ndarray
    rays_lengths: np.ndarray


class Renderer:

    def __init__(self, env_data: data.EnvData):
        self.window_width = env_data.config.window_width
        self.window_height = env_data.config.window_height
        self.geom_list = self._init_render_objects(env_data)
        self.viewer = rendering.Viewer(self.window_width, self.window_height)
        self.buffers = [
            env_data.shared_buffers.dogs_positions,
            env_data.shared_buffers.sheep_positions,
            env_data.shared_buffers.target_position,
            env_data.shared_buffers.observation,
            env_data.shared_buffers.rays_lengths
        ]
        for geom in self.geom_list:
            self.viewer.geoms.extend(geom.get_parts())

    @staticmethod
    def _init_render_objects(env_data):
        geom_list = []
        dogs_count = env_data.config.dogs_count
        sheep_count = env_data.config.sheep_count

        for i in range(dogs_count):
            geom_list.append(dog_geom.DogGeom(env_data, i))

        for i in range(sheep_count):
            geom_list.append(sheep_geom.SheepGeom(env_data, i))

        geom_list.append(target_geom.Target())

        return geom_list

    def render(self):
        arrays = Arrays(*[buffer.map_read() for buffer in self.buffers])
        for geom in self.geom_list:
            geom.update(arrays)

        self.viewer.render()
        for buffer in self.buffers:
            buffer.unmap()

    def close(self):
        self.viewer.close()
