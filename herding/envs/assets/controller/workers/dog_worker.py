from herding.envs.assets.multiprocessing import Worker
from herding.envs.assets.agents import Dog


class DogWorker(Worker):

    def __init__(self, shared_data, dog_params, dogs_range):
        super().__init__()

        self.shared_data = shared_data
        self.dog_params = dog_params
        self.dogs_range = dogs_range
        self.dogs_list = []

        self.create_dogs()

    def create_dogs(self):
        for i in self.dogs_range:
            self.dogs_list.append(Dog(i, *self.dog_params))

    def move(self):
        pass

    def update_observation(self):
        pass
