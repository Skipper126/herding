import itertools

from herding.rendering.env_buffers_mapper import EnvArraysMapper
from herding import data
import pygame

from herding.rendering.geoms.dog_geom import DogGeom
from herding.rendering.geoms.sheep_geom import SheepGeom


class Renderer:

    def __init__(self, env_data: data.EnvData):
        self.window_width = env_data.config.window_width
        self.window_height = env_data.config.window_height
        self.agents_matrix_side_length = env_data.config.agents_matrix_side_length
        self.env_arrays_mapper = EnvArraysMapper(env_data)

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        self.sheep_geom = SheepGeom(env_data, self.screen)
        self.dog_geom = DogGeom(env_data, self.screen)

    def render(self):
        arrays = self.env_arrays_mapper.map_env_arrays()
        self.screen.fill('white')

        for i, j in itertools.product(range(self.agents_matrix_side_length), range(self.agents_matrix_side_length)):
            agent = arrays.input_matrix[i, j]
            if agent[4] == 0:
                self.sheep_geom.draw(agent, i, j)
            else:
                self.dog_geom.draw(agent, arrays.observation, i, j)

        pygame.display.flip()

        self.env_arrays_mapper.unmap_env_arrays()

    def set_text(self, text: str):
        pass

    def close(self):
        pass
