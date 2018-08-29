from .geom import *
from gym.envs.classic_control import rendering
import math


class DogGeom(Geom):

    COLOR = {
        -1: (1, 0, 0),
        0: (0.5, 0.5, 0.5),
        1: (0, 1, 0)
    }

    def __init__(self, env_data, dog_index):
        self.dog_radius = env_data.config.agent_radius
        self.rays_count = env_data.config.rays_count
        self.ray_length = env_data.config.ray_length
        self.dog_pos = env_data.dogs_positions[dog_index]
        self.dog_rotation = env_data.dogs_rotations[dog_index]
        self.dog_rays = env_data.rays[dog_index]

        self.body = Part(rendering.make_circle(self.dog_radius, res=8))
        self.body.set_color(185 / 255, 14 / 255, 37 / 255)
        self.rays = []
        for _ in range(self.rays_count):
            self.rays.append(Part(rendering.Line((0, 0), (self.ray_length, 0))))

    def get_parts(self):
        parts = [self.body.body]
        for ray in self.rays:
            parts.append(ray.body)
        return parts

    def update(self):
        self.body.set_pos(self.dog_pos[0], self.dog_pos[1])
        for i, ray in enumerate(self.rays):
            ray.set_scale(1 - self.dog_rays[i, 0], 0)
            color = tuple(min(x * (1.5 - self.dog_rays[i, 0]), 1) for x in self.COLOR[self.dog_rays[i, 1]])
            ray.set_color(*color)
            # TODO check the ray_radian
            rot = self.dog_rotation - ((math.pi / 180) * i) # - self.object.ray_radian[i]
            ray.set_rotation(rot)
            x = math.cos(rot) * self.dog_radius
            y = math.sin(rot) * self.dog_radius
            ray.set_pos(self.dog_pos[0] + x, self.dog_pos[1] + y)
