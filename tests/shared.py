from herding import data


def get_env_data(params) -> data.EnvData:
    env_data = data.EnvData(config=data.create_config(params))
    data.init_opencl(env_data)

    return env_data
