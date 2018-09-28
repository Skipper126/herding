from herding.data.env_data import EnvData, Config
from herding.data.factory import create_env_data
from herding.data.configuration import get_color_tuple_from_config
import os


def get_env_data_header_path() -> str:
    path = os.path.join(os.path.dirname(__file__), 'env_data.cuh')

    return path
