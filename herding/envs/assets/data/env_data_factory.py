import os
import json
from .env_data import EnvData

def create_env_data(params) -> EnvData:
    env_data = EnvData()
    config = _create_config(params)
    env_data.config = config
    env_data.observation = _create_observation_array(config)
    env_data.dogs_positions = _create_dogs_positions_array(config)
    env_data.sheep_positions = _create_sheep_positions_array(config)
    env_data.dogs_rotations = _create_dogs_rotations_array(config)
    env_data.herd_centre = _create_herd_centre_array(config)

    return env_data

def _create_config(params):
    config = _get_default_configuration()
    config.update(params)

    return config

def _create_dogs_positions_array(config):
    pass

def _create_dogs_rotations_array(config):
    pass

def _create_sheep_positions_array(config):
    pass

def _create_observation_array(config):
    pass

def _create_herd_centre_array(config):
    pass

def _get_default_configuration():
    config = {}
    conf_dir = os.path.join(os.path.dirname(__file__), 'configuration.json')
    with open(conf_dir) as json_file:
        json_config = json.load(json_file)
        config.update(json_config)
    _convert_bool(config)

    return config

def _convert_bool(config):
    for key, value in config.items():
        if type(value) is str:
            if value.capitalize() == 'True':
                config[key] = True
            elif value.capitalize() == 'False':
                config[key] = False