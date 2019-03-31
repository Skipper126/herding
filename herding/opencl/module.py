import pyopencl as cl


class Module:

    def __init__(self, queue, prg, function, args):
        self.queue = queue
        self.prg = prg
        self.kernel = getattr(prg, function)
        self.kernel.set_args(*args)

    def run(self, global_size, local_size):
        cl.enqueue_nd_range_kernel(self.queue, self.kernel, global_size, local_size)
