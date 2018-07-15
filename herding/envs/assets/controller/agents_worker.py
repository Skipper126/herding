from herding.envs.assets.multiprocessing import Worker
from herding.envs.assets.agents import Dog, Sheep


class AgentsWorker(Worker):

    def __init__(self, env_data, dogs_ranges, sheep_ranges):
        super().__init__()

        self.env_data = env_data
        self.dogs_ranges = dogs_ranges
        self.sheep_ranges = sheep_ranges
        self.dogs_list, \
        self.sheep_list = self._create_agents()

    def _create_agents(self):
        dogs_list = []
        sheep_list = []

        for i in self.dogs_ranges:
            dogs_list.append(Dog(self.env_data, i))

        for i in self.sheep_ranges:
            sheep_list.append(Sheep(self.env_data, i))

        return dogs_list, sheep_list

    def move_dogs(self, action):
        for i, dog in enumerate(self.dogs_list):
            dog.move(action[i])

    def move_sheep(self):
        for sheep in self.sheep_list:
            sheep.move()

    def update_observation(self):
        for dog in self.dogs_list:
            dog.update_observation()

