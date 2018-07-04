
def start(pipe, worker_type, worker_args):
    worker = worker_type(*worker_args)

    while not worker.should_quit():
        task = pipe.recv()
        getattr(worker, task[0])(*task[1])
        pipe.send(0)


class Worker:

    def __init__(self):
        self.__quit = False

    def quit(self):
        self.__quit = True

    def should_quit(self):
        return self.__quit


