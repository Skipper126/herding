from herding.envs.assets.multiprocessing import SharedNumpy
from herding.envs.assets.configuration.constants import ConfigName as cn


class SharedData:

    def __init__(self, config):

        self._dogs_positions = SharedNumpy((config[cn.DOGS_COUNT], 2))
        self._sheep_positions = SharedNumpy((config[cn.SHEEP_COUNT], 2))
        self._herd_centre = SharedNumpy((2,))
        self._observation = SharedNumpy((config[cn.DOGS_COUNT], config[cn.RAYS_COUNT], 2))

    @property
    def dogs_positions(self):
        return self._dogs_positions.get()

    @property
    def sheep_positions(self):
        return self._sheep_positions.get()

    @property
    def herd_centre(self):
        return self._herd_centre.get()

    @property
    def observation(self):
        return self._observation.get()
