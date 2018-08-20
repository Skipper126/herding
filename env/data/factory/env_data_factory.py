from env.data.env_data import EnvData
from env.data.factory import host, device


def create_env_data(params) -> EnvData:
    host_env_data = host.get_host_env_data(params)
    device_env_data = device.get_gpu_env_data(host_env_data)

    env_data = EnvData(**host_env_data.update(device_env_data))

    return env_data


