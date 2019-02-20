from .geom import *
from gym.envs.classic_control import rendering
import math


class DogGeom(Geom):

    def __init__(self, env_data, dog_index):
        self.agent_radius = env_data.config.agent_radius
        self.rays_count = env_data.config.rays_count
        self.ray_length = env_data.config.ray_length
        self.position = env_data.host_arrays.dogs_positions[dog_index]
        self.rotation = env_data.host_arrays.dogs_rotations[dog_index]
        self.rays_lengths = env_data.host_arrays.rays_lengths[dog_index]
        self.observation = env_data.host_arrays.observation[dog_index]

        self.body = Part(rendering.make_circle(self.agent_radius, res=8))
        self.body.set_color(0.7, 0.1, 0.2)
        self.rays = []
        for _ in range(self.rays_count):
            self.rays.append(Part(rendering.Line((0, 0), (self.ray_length - self.agent_radius, 0))))

    def get_parts(self):
        parts = [self.body.body]
        for ray in self.rays:
            parts.append(ray.body)
        return parts

    def update(self):
        self.body.set_pos(self.position[0], self.position[1])
        for i, ray in enumerate(self.rays):
            ray.set_scale(self.rays_lengths[i], 0)
            color = tuple(self.observation[i])
            ray.set_color(*color)
            rot = self.rotation + (i / self.rays_count) * math.pi
            ray.set_rotation(rot)
            x = math.cos(rot) * self.agent_radius
            y = math.sin(rot) * self.agent_radius
            ray.set_pos(self.position[0] + x, self.position[1] + y)
