import numpy as np
from typing import NamedTuple, List, Dict


class ArrayInfo(NamedTuple):
    name: str
    shape: List[int]
    size: int
    offset: int

def get_arrays_info(arrays_shapes: Dict[str, List[int]]) -> List[ArrayInfo]:

    arrays_info = []
    offset = 0
    for name, shape in arrays_shapes.items():
        size = _get_array_size(shape)
        arrays_info.append(ArrayInfo(
            name=name,
            shape=shape,
            size=size,
            offset=offset,
        ))
        offset += size * 4

    return arrays_info


def _get_array_size(shape):
    return int(np.prod(np.array(list(shape))))
