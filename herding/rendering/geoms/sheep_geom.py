import pygame as pygame
import numpy as np
from ...data import EnvData


class SheepGeom():

    def __init__(self, env_data: EnvData, screen: pygame.Surface):
        self.screen = screen

        self.agent_radius = env_data.config.agent_radius
        self.surf = pygame.Surface((5, 5), pygame.SRCALPHA)
        self.font = pygame.font.SysFont('Arial', 15)
        self.rect = self.surf.get_rect()
        self.surf.set_colorkey('white')
        self.surf.fill('white')
        #pygame.draw.rect(self.surf, 'grey', self.rect)

    def draw(self, agent: np.ndarray, i, j):
        pos_x = agent[0]
        pos_y = agent[1]

        self.rect.x = pos_x - self.agent_radius
        self.rect.y = pos_y - self.agent_radius

        color = 'grey' if agent[5] == 0 else 'blue'
        pygame.draw.rect(self.surf, color, (0, 0, 5, 5))
        # self.surf.fill('white')
        # pygame.draw.circle(self.surf, 'grey', (self.agent_radius, self.agent_radius), self.agent_radius)
        # textsurf = self.font.render(f'{i}, {j}', False, (0, 0, 0))
        # self.surf.blit(textsurf, (0, 0))

        self.screen.blit(self.surf, self.rect)
