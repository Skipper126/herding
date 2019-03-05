import pyopencl as cl


class OpenClModule:

    def __init__(self, ocl, prg, buffer):
        self.queue = ocl.queue
        self.prg = prg
        # assume only one kernel per module
        self.kernel = prg.all_kernels()[0]
        self.kernel.set_arg(0, buffer)

    def run(self, nd_range):
        cl.enqueue_nd_range_kernel(self.queue, self.kernel, nd_range, nd_range)
