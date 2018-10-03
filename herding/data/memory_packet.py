import numpy as np
from typing import List
from herding.data import EnvData
from herding.data.configuration import get_arrays_shapes
from herding.data.factory.info import get_arrays_info

def get_memory_packet(env_data: EnvData, packet_names: List[str], device_only_arrays=False) -> np.ndarray:
    env_data_names = _get_arrays_names_list_from_env_data(env_data)
    fit_index = _get_fit_index(packet_names, env_data_names)
    if _check_fit(packet_names, env_data_names, fit_index) is False:
        raise Exception('Packet arrays don\'t fit env_data.')
    arrays_info = get_arrays_info(get_arrays_shapes(env_data.config)[''])






def _get_arrays_from_names(env_data, packet_names):
    arrays = []
    for packet_name in packet_names:
        arrays.append(getattr(env_data, packet_name))

    return arrays

def _check_fit(packet_names, env_data_names, fit_index):
    fits = True
    for packet_name in packet_names:
        if packet_name != env_data_names[fit_index]:
            fits = False
            break
        else:
            fit_index += 1

    return fits

def _get_fit_index(packet_names, env_data_names):
    fit_index = -1
    for i in range(len(env_data_names)):
        if env_data_names[i] == packet_names[0]:
            fit_index = i
            break
    if fit_index == -1:
        raise Exception('Packet names not found in env_data.')

    return fit_index

def _get_arrays_names_list_from_env_data(env_data):
    arrays_names = []
    for key, value in env_data.config._asdict():
        arrays_names.append(key)
    return arrays_names
