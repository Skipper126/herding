from gym.envs.classic_control import rendering
from herding.envs.assets.configuration.names import ConfigName as cn
from .geoms import *


class Renderer:

    def __init__(self, env_data):
        self.map_width = env_data.config[cn.MAP_WIDTH]
        self.map_height = env_data.config[cn.MAP_HEIGHT]
        self.geom_list = self._initRenderObjects(env_data)
        self.viewer = rendering.Viewer(self.map_width, self.map_height)

        for geom in self.geom_list:
            self.viewer.geoms.extend(geom.get_parts())

    def _initRenderObjects(self, env):
        geom_list = []

        for dog in env.dog_list:
            geom_list.append(dog_geom.DogGeom(dog))

        for sheep in env.sheep_list:
            geom_list.append(sheep_geom.SheepGeom(sheep))

        geom_list.append(crosshair.Crosshair(env))

        return geom_list

    def render(self):
        for geom in self.geom_list:
            geom.update()

        self.viewer.render()

    def close(self):
        self.viewer.close()
