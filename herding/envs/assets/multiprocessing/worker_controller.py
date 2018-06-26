import multiprocessing as mp


class WorkerController:

    def __init__(self, WorkerType):
        self.pipe, child_pipe = mp.Pipe()
        self.process = mp.Process(target=WorkerType.start, args=child_pipe)

    def execute(self, method, args=None):
        self.pipe.send((method,args))

    def start(self):
        self.process.start()

    def join(self):
        self.process.join()
