import multiprocessing as mp
from herding.envs.assets.multiprocessing.worker_starter import start_worker


class WorkerController:

    def __init__(self, worker_type, worker_args):
        self.pipe, child_pipe = mp.Pipe()
        self.process = mp.Process(target=start_worker, args=(child_pipe, worker_type, worker_args))

    def execute(self, method, args=()):
        self.pipe.send((method, args))

    def start(self):
        self.process.start()

    def join(self):
        self.process.join()

    def wait(self):
        self.pipe.recv()
