import numpy as np
import math
import pygame

from ...data import EnvData


class DogGeom():

    def __init__(self, env_data: EnvData, screen: pygame.Surface):
        self.agent_radius = env_data.config.agent_radius
        self.rays_count = env_data.config.rays_count
        self.ray_length = env_data.config.ray_length

        self.screen = screen
        self.font = pygame.font.SysFont('Arial', 15)
        self.surface_size = self.agent_radius * 2 + self.ray_length * 2
        self.surf = pygame.Surface((self.surface_size, self.surface_size), pygame.SRCALPHA)
        self.rect = self.surf.get_rect()
        self.surf.set_colorkey('white')
        self.surf.fill('white')


    def draw(self, agent: np.ndarray, observations: np.ndarray, i, j):
        pos_x = agent[0]
        pos_y = agent[1]
        rotation = agent[2]
        radius = 3
        dir_x = int(round(math.cos(rotation)) * radius)
        dir_y = -1 * int(round(math.sin(rotation)) * radius)
        observation = observations[int(agent[5])]

        self.rect.x = pos_x - self.surface_size / 2
        self.rect.y = pos_y - self.surface_size / 2

        self.surf.fill('white')
        pygame.draw.circle(self.surf, 'red', (self.surface_size / 2, self.surface_size / 2), self.agent_radius)
        textsurf = self.font.render(f'{dir_x}, {dir_y}', False, (0, 0, 0))

        for ray_id in range(self.rays_count):
            ray_angle = rotation + ray_id * math.pi / self.rays_count
            x1 = (math.cos(ray_angle) * self.agent_radius) + (self.surface_size / 2)
            y1 = (math.sin(ray_angle) * self.agent_radius) + (self.surface_size / 2)

            x2 = x1 + (self.ray_length - self.agent_radius) * math.cos(ray_angle)
            y2 = y1 + (self.ray_length - self.agent_radius) * math.sin(ray_angle)
            pygame.draw.aaline(self.surf, tuple(observation[ray_id] * 255), (x1, y1), (x2, y2))

        self.surf.blit(textsurf, (self.surface_size / 2 - 10, self.surface_size / 2 - 10))
        self.screen.blit(self.surf, self.rect)

