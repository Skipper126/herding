import pygame
from gym.envs.classic_control import rendering
import numpy as np
from herding.data import EnvData


class Target():

    def __init__(self, env_data: EnvData, screen: pygame.Surface):
        self.screen = screen
        self.pos_x = env_data.config.target_x
        self.pos_y = env_data.config.target_y
        self.agent_radius = env_data.config.agent_radius
        self.surf = pygame.Surface((self.agent_radius * 2, self.agent_radius * 2), pygame.SRCALPHA)
        self.rect = self.surf.get_rect()
        self.rect.x = self.pos_x
        self.rect.y = self.pos_y
        self.surf.set_colorkey('white')
        self.surf.fill('white')
        pygame.draw.circle(self.surf, 'blue', (self.agent_radius, self.agent_radius), self.agent_radius)

    def draw(self):
        self.screen.blit(self.surf, self.rect)
