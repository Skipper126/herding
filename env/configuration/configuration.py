import json
import os


def get_default_configuration():
    config = {}
    conf_dir = os.path.join(os.path.dirname(__file__), 'configuration.json')
    with open(conf_dir) as json_file:
        json_config = json.load(json_file)
        config.update(json_config)

    return config
