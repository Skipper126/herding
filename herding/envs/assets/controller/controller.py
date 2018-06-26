import multiprocessing as mp
from herding.envs.assets.multiprocessing import WorkerController
from herding.envs.assets.controller.workers import dog_worker, sheep_worker


class Controller:

    def __init__(self, env):
        self.dog_workers = []
        self.sheep_workers = []
        self.workers_count = mp.cpu_count()
        self.dog_workers_count, \
        self.sheep_workers_count = self._distribute_resources_over_workers()
        self._create_workers()
        self._set_workers_args(env)

    def move_dogs(self, action):
        for worker in self.dog_workers:
            worker.execute('move_dogs', action)

    def move_sheep(self):
        for worker in self.sheep_workers:
            worker.execute('move')

    def get_observation(self):
        for worker in self.dog_workers:
            worker.execute('get_observation')

    def close(self):
        self._quit_workers()

    def _create_workers(self):
        for i in range(self.dog_workers_count):
            worker = WorkerController(dog_worker.DogWorker)
            self.dog_workers.append(worker)

        for i in range(self.sheep_workers_count):
            worker = WorkerController(sheep_worker.SheepWorker)
            self.sheep_workers.append(worker)

    def _start_workers(self):
        for worker in self.dog_workers + self.sheep_workers:
            worker.start()

    def _quit_workers(self):
        for worker in self.dog_workers + self.sheep_workers:
            worker.execute('quit')
            worker.join()

    def _distribute_resources_over_workers(self):
        #TODO implement logic to distribute resources
        return 1, 1

    def _set_workers_args(self, env):
        pass
