from .geom import *
from gym.envs.classic_control import rendering
from herding.data import get_color_tuple_from_config
import math


class DogGeom(Geom):

    def __init__(self, env_data, dog_index):
        self.dog_radius = env_data.config.agent_radius
        self.rays_count = env_data.config.rays_count
        self.ray_length = env_data.config.ray_length
        self.dog_pos = env_data.dogs_positions[dog_index]
        self.dog_rotation = env_data.dogs_rotations[dog_index]
        self.dog_rays_lengths = env_data.rays_lengths[dog_index]
        self.dog_observation = env_data.observation[dog_index]

        self.body = Part(rendering.make_circle(self.dog_radius, res=8))
        #dog_color = get_color_tuple_from_config(env_data.config.dog_color)
        self.body.set_color(*(env_data.config.dog_color_r,
                              env_data.config.dog_color_g,
                              env_data.config.dog_color_b,))
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
            ray.set_scale(self.dog_rays_lengths[i], 0)
            color = tuple(self.dog_observation[i][0])
            ray.set_color(*color)
            # TODO check the ray_radian
            rot = self.dog_rotation + (i / 128) * math.pi# - self.object.ray_radian[i]
            ray.set_rotation(rot)
            x = math.cos(rot) * self.dog_radius
            y = math.sin(rot) * self.dog_radius
            ray.set_pos(self.dog_pos[0] + x, self.dog_pos[1] + y)
