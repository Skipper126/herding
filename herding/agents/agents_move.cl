#include "herding/data/env_data.h"
#include "herding/agents/kernels/declarations.h"

__kernel void move_agents(Arrays *arrays)
{
    if (threadIdx.x < DOGS_COUNT)
    {
        move_dogs(arrays);
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (threadIdx.x < SHEEP_COUNT)
    {
        move_sheep_simple(arrays);
    }
}
