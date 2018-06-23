from ..dog.dog import Dog
from .. import agents_shared_state
from multiprocessing import Pipe

CREATE_AGENTS, \
UPDATE_OBSERVATION, \
QUIT = range(3)

class DogWorker:

    def start(self, pipe):

        self.quit = False
        pipe = pipe
        tasks = {
            CREATE_AGENTS: self.create_dogs,
            UPDATE_OBSERVATION: self.update_observation,
            QUIT
        }

        while not quit:
            task = pipe.recv()


    def create_dogs(self):
        pass

    def move(self):
        pass

    def update_observation(self):
        pass

def start_dogs(pipe, mp_dogs_positions, mp_dogs_done, start_index, worker_dog_count, all_dog_count):
    dogs_positions = agents_shared_state.mp_array_to_numpy_array(mp_dogs_positions, (all_dog_count, 2))
    dogs_done = agents_shared_state.mp_array_to_numpy_array(mp_dogs_done, (all_dog_count, 2))


    quit = False
    while not quit:
