from herding.rendering.env_buffers_mapper import EnvArraysMapper
from herding import data
import pygame

from herding.rendering.sprites.dog_geom import DogGeom
from herding.rendering.sprites.sheep_geom import SheepGeom


class Renderer:

    def __init__(self, env_data: data.EnvData):
        self.window_width = env_data.config.window_width
        self.window_height = env_data.config.window_height
        self.env_arrays_mapper = EnvArraysMapper(env_data)


        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)

        self.sheep_geoms = [SheepGeom(env_data, self.screen, index) for index in range(env_data.config.sheep_count)]
        self.dogs_geoms = [DogGeom(env_data, self.screen, index) for index in range(env_data.config.dogs_count)]

    def render(self):
        arrays = self.env_arrays_mapper.map_env_arrays()
        self.screen.fill('white')

        for sheep in self.sheep_geoms:
            sheep.update(arrays)

        for dog in self.dogs_geoms:
            dog.update(arrays)

        pygame.display.flip()

        self.env_arrays_mapper.unmap_env_arrays()

    def set_text(self, text: str):
        pass

    def close(self):
        pass
