import multiprocessing as mp
from herding.envs.assets.multiprocessing import WorkerController
from herding.envs.assets.controller.workers import dog_worker, sheep_worker
import math


class Controller:

    def __init__(self, env):
        self.dog_workers = []
        self.sheep_workers = []
        self.workers_count = mp.cpu_count()
        self.dog_workers_count, \
        self.sheep_workers_count = self._get_workers_distribution()
        self._create_workers(env)

    def move_dogs(self, action):
        self._execute_and_wait(self.dog_workers, 'move_dogs', action)

    def move_sheep(self):
        self._execute_and_wait(self.sheep_workers, 'move_sheep')

    def get_observation(self):
        self._execute_and_wait(self.dog_workers, 'get_observations')

    def close(self):
        self._quit_workers()

    def _create_workers(self, env):
        for i in range(self.dog_workers_count):
            worker = WorkerController(dog_worker.DogWorker, self._get_worker_args(env))
            self.dog_workers.append(worker)

        for i in range(self.sheep_workers_count):
            worker = WorkerController(sheep_worker.SheepWorker, self._get_worker_args(env))
            self.sheep_workers.append(worker)

    def _start_workers(self):
        for worker in self.dog_workers + self.sheep_workers:
            worker.start()

    def _quit_workers(self):
        for worker in self.dog_workers + self.sheep_workers:
            worker.execute('quit')
            worker.wait()
            worker.join()

    def _execute_and_wait(self, workers, method, args=()):
        for worker in workers:
            worker.execute(method, args)

        for worker in workers:
            worker.wait()

    def _get_workers_distribution(self):
        # TODO implement logic to distribute resources
        return int(math.ceil(self.workers_count * 0.8)), int(math.floor(self.workers_count * 0.2))

    def _get_worker_args(self, env):
        return env.params, env.shared_data
