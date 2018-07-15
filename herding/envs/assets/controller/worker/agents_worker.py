from herding.envs.assets.multiprocessing import Worker
from herding.envs.assets.agents import Dog, Sheep
from herding.envs.assets.configuration import constants as c

class AgentsWorker(Worker):

    def __init__(self, env_data, worker_ranges):
        super().__init__()

        self.dogs_list = []
        self.sheep_list = []

        self.shared_data = env_data.shared_data
        self.worker_ranges = worker_ranges
        self.dogs_list = self._create_dogs()

    def _create_agents(self, type):
        dogs_list = []
        for i in self.worker_ranges:
            dogs_list.append(Dog(self.params, self.shared_data, i))

        return dogs_list

    def move(self, action):
        for i, dog in enumerate(self.dogs_list):
            dog.move(action[i])

    def update_observation(self):
        for dog in self.dogs_list:
            dog.update_observation()
