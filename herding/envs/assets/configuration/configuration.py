import json
import os


def get_configuration():
    config = {}
    with open('configuration.json') as json_file:
        json_config = json.load(json_file)
        config.update(json_config)

    return config
