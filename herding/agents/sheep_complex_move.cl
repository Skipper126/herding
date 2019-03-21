#include <curand.h>

__device__ void move_sheep_complex(Arrays *arrays)
{
    move_sheep_simple(arrays);
}
