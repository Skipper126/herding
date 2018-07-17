from herding.envs.assets.multiprocessing import WorkerController
from herding.envs.assets.configuration.names import ConfigName as cn
from .agents_workers_utils import get_workers_count, get_workers_ranges
from .agents_worker import AgentsWorker


class AgentsController:

    def __init__(self, env_data):
        self.workers = []
        workers_count = get_workers_count(env_data)
        dogs_count = env_data.config[cn.DOGS_COUNT]
        sheep_count = env_data.config[cn.SHEEP_COUNT]
        dogs_workers_ranges = get_workers_ranges(dogs_count, workers_count)
        sheep_workers_ranges = get_workers_ranges(sheep_count, workers_count)

        for i in range(workers_count):
            self.workers.append(WorkerController(AgentsWorker, (env_data,
                                                                dogs_workers_ranges,
                                                                sheep_workers_ranges)))

        self.dogs_workers_ranges = dogs_workers_ranges

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
