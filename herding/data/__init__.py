from herding.data.env_data import EnvData, Config
from herding.data.factory import create_env_data
from herding.data.memory_buffer import MemoryBuffer, get_memory_buffer
import os


def get_env_data_header_path() -> str:
    path = os.path.join(os.path.dirname(__file__), 'env_data.cuh')

    return path
