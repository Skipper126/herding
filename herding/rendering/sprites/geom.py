from herding.rendering.env_buffers_mapper import EnvArrays


class Geom:

    def update(self, env_arrays: EnvArrays):
        raise NotImplementedError
