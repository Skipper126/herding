from ..agents_controller import AgentsController
from . import functions
from . import cn
class CpuAgentsController(AgentsController):

    def __init__(self, env_data):
        self.dog_movement = self._get_dog_movement_controller(env_data)
        self.sheep_movement = self._get_sheep_movement_controller(env_data)

    def move_agents(self, action):
        for i in dogs_count:
            functions.move_dog(i, action[i])
        for i in sheep_count:
            functions.move_sheep(i)

    def get_observation(self):
        for i in dogs_count:
            functions.update_observation()

        return self.observation

    def close(self):
        pass

    @staticmethod
    def _get_dog_movement_controller(env_data):
        if rotation is 'free':
            return DogMoveFree(env_data)
        else:
            return DogMoveCentered(env_data)

    @staticmethod
    def _get_sheep_movement_controller(env_data):
        if type is 'simple':
            return SheepMoveSimple(env_data)
        else:
            return SheepMoveComplex(env_data)