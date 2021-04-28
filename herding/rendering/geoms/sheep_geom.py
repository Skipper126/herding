import pygame as pygame
import numpy as np
from ...data import EnvData


class SheepGeom():

    def __init__(self, env_data: EnvData, screen: pygame.Surface):
        self.screen = screen

        self.agent_radius = env_data.config.agent_radius
        self.surf = pygame.Surface((self.agent_radius * 2, self.agent_radius * 2), pygame.SRCALPHA)
        self.rect = self.surf.get_rect()
        self.surf.set_colorkey('white')
        self.surf.fill('white')
        pygame.draw.circle(self.surf, 'grey', (self.agent_radius, self.agent_radius), self.agent_radius)


    def draw(self, agent: np.ndarray):
        pos_x = agent[0]
        pos_y = agent[1]

        self.rect.x = pos_x - self.agent_radius
        self.rect.y = pos_y - self.agent_radius
        self.screen.blit(self.surf, self.rect)
