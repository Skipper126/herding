from herding.envs.assets.multiprocessing import WorkerController
from . import agents_workers_factory
import math


class AgentsController:

    def __init__(self, env_data):
        self.workers, \
        self.dogs_workers_ranges = agents_workers_factory.get_workers(env_data)

    def move_dogs(self, action):
        for i, worker in enumerate(self.workers):
            action_slice = [action[j] for j in self.dogs_workers_ranges[i]]
            worker.execute('move_dogs', (action_slice,))

        for worker in self.workers:
            worker.wait()

    def move_sheep(self):
        self._execute_and_wait(self.workers, 'move_sheep')

    def get_observation(self):
        self._execute_and_wait(self.workers, 'update_observations')

    def close(self):
        self._quit_workers()

    def _start_workers(self):
        for worker in self.workers:
            worker.start()

    def _quit_workers(self):
        for worker in self.workers:
            worker.execute('quit')
            worker.wait()
            worker.join()

    def _execute_and_wait(self, method, args=()):
        for worker in self.workers:
            worker.execute(method, args)

        self._wait_for_workers()

    def _wait_for_workers(self):
        for worker in self.workers:
            worker.wait()
