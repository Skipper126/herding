import json
import os
from typing import Dict, List
from string import Template
from herding.data.env_data import Config, EnvData


def get_default_configuration() -> Dict:
    config = {}
    conf_path = os.path.join(os.path.dirname(__file__), 'configuration.json')
    with open(conf_path) as json_file:
        json_config = json.load(json_file)
        config.update(json_config)

    return config


def get_arrays_shapes(config: Config) -> Dict:

    path = os.path.join(os.path.dirname(__file__), 'arrays.json')
    with open(path, 'r') as file:
        raw_content = file.read()

    config_dict = config._asdict()
    config_dict['entities_count'] = config.dogs_count + config.sheep_count + 1
    content = Template(raw_content).substitute(config_dict)
    arrays_shapes = json.loads(content)

    return arrays_shapes


_ocl_defines = [
    'dogs_count',
    'sheep_count',
    'herd_target_radius',
    'agent_radius',
    'max_movement_speed',
    'max_rotation_speed',
    'max_episode_reward',
    'rays_count',
    'ray_length',
    'field_of_view',
    'agents_layout_range',
    'channels_count'
]


def get_ocl_defines(env_data: EnvData) -> List[str]:
    processed_defines = [Template(
        '-D ' + name.upper() + '=$' + name
    ).substitute(env_data.config._asdict())
                         for name in _ocl_defines]
    return processed_defines
