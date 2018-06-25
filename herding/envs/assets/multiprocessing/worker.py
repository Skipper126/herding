import multiprocessing as mp


class Worker:

    @staticmethod
    def start(pipe):
        worker = Worker()

        while not worker.done:
            task = pipe.recv()
            if task[1] is not None:
                getattr(worker, task[0])(*task[1])
            else:
                getattr(worker, task[0])()

    def __init__(self):
        self.done = False

    def quit(self):
        self.done = True


