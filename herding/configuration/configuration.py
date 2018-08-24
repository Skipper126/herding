import json
import os
from typing import Dict


def get_default_configuration() -> Dict:
    config = {}
    conf_path = os.path.join(os.path.dirname(__file__), 'configuration.json')
    with open(conf_path) as json_file:
        json_config = json.load(json_file)
        config.update(json_config)

    return config
