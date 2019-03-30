import pyopencl as cl


class Module:

    def __init__(self, queue, prg, function, args):
        self.queue = queue
        self.prg = prg
        self.kernel = getattr(prg, function)
        self.kernel.set_args(args)

    def run(self, nd_range):
        cl.enqueue_nd_range_kernel(self.queue, self.kernel, (nd_range,), (nd_range,))
