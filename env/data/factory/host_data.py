from env import configuration


def create_config(params):
    config = configuration.get_default_configuration()
    config.update(params)

    return config


def create_dogs_positions_array(config):
    return 0


def create_dogs_rotations_array(config):
    return 0


def create_sheep_positions_array(config):
    return 0


def create_observation_array(config):
    return 0


def create_herd_centre_array(config):
    return 0