from herding.envs.assets.multiprocessing import Worker
from herding.envs.assets.agents import Dog


class DogWorker(Worker):

    def __init__(self, params, shared_data, worker_range):
        super().__init__()

        self.shared_data = shared_data
        self.params = params
        self.worker_range = worker_range
        self.dogs_list = self._create_dogs()

    def _create_dogs(self):
        dogs_list = []
        for i in self.worker_range:
            dogs_list.append(Dog(self.params, self.shared_data, i))

        return dogs_list

    def move(self, action):
        for i, dog in enumerate(self.dogs_list):
            dog.move(action[i])

    def update_observation(self):
        for dog in self.dogs_list:
            dog.update_observation()
