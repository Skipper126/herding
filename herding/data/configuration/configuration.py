import json
import os
from typing import Dict
from herding.data.env_data import Config
from string import Template

def get_default_configuration() -> Dict:
    config = {}
    conf_path = os.path.join(os.path.dirname(__file__), 'configuration.json')
    with open(conf_path) as json_file:
        json_config = json.load(json_file)
        config.update(json_config)

    return config

def get_arrays_shapes(config: Config) -> Dict:

    path = os.path.join(os.path.dirname(__file__), 'arrays_shapes.json')
    with open(path, 'r') as file:
        raw_content = file.read()
    content = Template(raw_content).substitute(config._asdict())
    arrays_shapes = json.loads(content)

    return arrays_shapes

