
def start(pipe, worker_type, worker_args):
    worker = worker_type(*worker_args)

    while not worker.is_done():
        task = pipe.recv()
        getattr(worker, task[0])(*task[1])


class Worker:

    def __init__(self):
        self.__done = False

    def quit(self):
        self.__done = True

    def is_done(self):
        return self.__done


