import pygame as pygame

from .geom import *
from gym.envs.classic_control import rendering

from ..env_buffers_mapper import EnvArrays
from ...data import EnvData


class SheepGeom(Geom):

    def __init__(self, env_data: EnvData, screen: pygame.Surface, index: int):
        self.index = index
        self.screen = screen

        self.agent_radius = env_data.config.agent_radius
        self.surf = pygame.Surface((self.agent_radius * 2, self.agent_radius * 2), pygame.SRCALPHA)
        self.rect = self.surf.get_rect()
        self.surf.set_colorkey('white')
        self.surf.fill('white')
        pygame.draw.circle(self.surf, 'grey', (self.agent_radius, self.agent_radius), self.agent_radius)


    def update(self, env_arrays: EnvArrays):
        self.rect.x = env_arrays.sheep_positions[self.index][0] - self.agent_radius
        self.rect.y = env_arrays.sheep_positions[self.index][1] - self.agent_radius
        self.screen.blit(self.surf, self.rect)
