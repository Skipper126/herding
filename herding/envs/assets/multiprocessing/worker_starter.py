
def start_worker(pipe, worker_type, worker_args):
    worker = worker_type(*worker_args)

    while not worker.should_quit():
        task = pipe.recv()
        getattr(worker, task[0])(*task[1])
        pipe.send(0)