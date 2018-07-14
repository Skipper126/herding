import multiprocessing as mp
from herding.envs.assets.multiprocessing import WorkerController
from herding.envs.assets.controller.workers import dog_worker, sheep_worker
import math


class AgentsController:

    def __init__(self, env):
        self.dog_workers = []
        self.sheep_workers = []
        self.workers_count = mp.cpu_count()
        self.dog_workers_count, \
        self.sheep_workers_count = self._get_workers_distribution()
        self.dogs_count = env.params.dog_count
        self.sheep_count = env.params.sheep_count

        self.dog_workers_ranges = self._get_workers_ranges(self.dogs_count, self.dog_workers_count)
        self.sheep_workers_ranges = self._get_workers_ranges(self.sheep_count, self.sheep_workers_count)

        self._create_workers(env)

    def move_dogs(self, action):
        for i, worker in enumerate(self.dog_workers):
            action_slice = [action[j] for j in self.dog_workers_ranges[i]]
            worker.execute('move_dogs', (action_slice,))

        for worker in self.dog_workers:
            worker.wait()

    def move_sheep(self):
        self._execute_and_wait(self.sheep_workers, 'move_sheep')

    def get_observation(self):
        self._execute_and_wait(self.dog_workers, 'get_observations')

    def close(self):
        self._quit_workers()

    def _create_workers(self, env):
        for i in range(self.dog_workers_count):
            worker = WorkerController(dog_worker.DogWorker, (env.params, env.shared_data, self.dog_workers_ranges[i]))
            self.dog_workers.append(worker)

        for i in range(self.sheep_workers_count):
            worker = WorkerController(sheep_worker.SheepWorker, (env.params, env.shared_data, self.sheep_workers_ranges[i]))
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

        self._wait_for_workers(workers)

    def _wait_for_workers(self, workers):
        for worker in workers:
            worker.wait()

    def _get_workers_distribution(self):
        # TODO implement logic to distribute resources
        return int(math.ceil(self.workers_count * 0.8)), int(math.floor(self.workers_count * 0.2))

    def _get_workers_ranges(self, agents_count, workers_count):
        workers_ranges = []

        j = 0
        for i in range(workers_count):
            workers_ranges.append([])
            for _ in range(int(math.ceil((agents_count) / workers_count))):
                if j < agents_count:
                    workers_ranges[i].append(j)
                    j += 1
                else:
                    break

        return workers_ranges

    def _get_worker_args(self, env):
        return env.params, env.shared_data
