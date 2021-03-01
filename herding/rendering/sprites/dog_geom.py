from .geom import *
from gym.envs.classic_control import rendering
import math
import pygame

from ...data import EnvData


class DogGeom(Geom):

    def __init__(self, env_data: EnvData, screen: pygame.Surface, index: int):
        self.agent_radius = env_data.config.agent_radius
        self.rays_count = env_data.config.rays_count
        self.ray_length = env_data.config.ray_length
        self.index = index

        self.screen = screen
        self.surface_size = self.agent_radius * 2 + self.ray_length * 2
        self.surf = pygame.Surface((self.surface_size, self.surface_size), pygame.SRCALPHA)
        self.rect = self.surf.get_rect()
        self.surf.set_colorkey('white')
        self.surf.fill('white')


    def update(self, arrays):
        self.rect.x = arrays.dogs_positions[self.index][0] - self.surface_size / 2
        self.rect.y = arrays.dogs_positions[self.index][1] - self.surface_size / 2

        self.surf.fill('white')
        pygame.draw.circle(self.surf, 'red', (self.surface_size / 2, self.surface_size / 2), self.agent_radius)

        for i in range (self.rays_count):
            rotation = arrays.dogs_positions[self.index][2] + (i / self.rays_count) * math.pi
            base_x = math.cos(rotation) * self.agent_radius + self.surface_size / 2
            base_y = math.sin(rotation) * self.agent_radius + self.surface_size / 2
            x1 = base_x
            y1 = base_y
            x2 = x1 + arrays.rays_lengths[self.index][i] * (self.ray_length - self.agent_radius * 3) * math.cos(rotation)
            y2 = y1 + arrays.rays_lengths[self.index][i] * (self.ray_length - self.agent_radius * 3) * math.sin(rotation)
            pygame.draw.aaline(self.surf, 'black', (x1, y1), (x2, y2))

        self.screen.blit(self.surf, self.rect)
