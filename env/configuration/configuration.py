import json
import os


def get_default_configuration():
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