from herding.data.env_data import EnvData, Config
from herding.data.factory import host, device
from herding.data import configuration


def create_env_data(params) -> EnvData:
    config = _get_config(params)
    arrays_shapes = configuration.get_arrays_shapes(config)
    host_buffer, host_arrays = host.get_host_arrays(arrays_shapes)
    device_buffer, device_arrays = device.get_device_arrays(arrays_shapes)
    env_data_params = {
        **{'config': config},
        **{'host_buffer': host_buffer},
        **{'device_buffer': device_buffer},
        **{'host_arrays': host_arrays},
        **{'device_arrays': device_arrays}
    }
    env_data = EnvData(**env_data_params)

    return env_data


def _get_config(params):
    config_dict = configuration.get_default_configuration()
    config_dict.update(params)
    config = Config(**config_dict)

    return config




