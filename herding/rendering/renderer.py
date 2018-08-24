from gym.envs.classic_control import rendering
from herding.rendering.geoms import *
from herding import data


class Renderer:

    def __init__(self, env_data: data.EnvData):
        self.map_width = env_data.config[env_data.config.map_width]
        self.map_height = env_data.config[env_data.config.map_height]
        self.geom_list = self._init_render_objects(env_data)
        self.viewer = rendering.Viewer(self.map_width, self.map_height)

        for geom in self.geom_list:
            self.viewer.geoms.extend(geom.get_parts())

    @staticmethod
    def _init_render_objects(env_data):
        geom_list = []
        dogs_count = env_data.config.dogs_count
        sheep_count = env_data.config.sheep_count

        for i in range(dogs_count):
            geom_list.append(dog_geom.DogGeom(env_data, i))

        for i in range(sheep_count):
            geom_list.append(sheep_geom.SheepGeom(env_data, i))

        geom_list.append(crosshair.Crosshair(env_data))

        return geom_list

    def render(self):
        for geom in self.geom_list:
            geom.update()

        self.viewer.render()

    def close(self):
        self.viewer.close()
