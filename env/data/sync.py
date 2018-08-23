from env import cuda


def sync_arrays(env_data):
    cuda.memcpy_dtoh(env_data.host_arrays, env_data.device_arrays)
