from .sheep_simple import SheepSimple
from .sheep_complex import SheepComplex
from .sheep_continuous import SheepContinuous

from herding.envs.assets.configuration.constants import SheepType

def get_sheep_class(sheep_type):
    return {
        SheepType.SIMPLE : SheepSimple,
        SheepType.COMPLEX : SheepComplex,
        SheepType.CONTINUOUS : SheepContinuous
    }[sheep_type]