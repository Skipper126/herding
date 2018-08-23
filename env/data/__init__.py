from env.data.env_data import EnvData, Config
from env.data.factory import create_env_data
import os


def get_env_data_header_path() -> str:
    path = os.path.join(os.path.dirname(__file__), 'env_data.cuh')

    return path
