#include "herding/opencl/rand.h"
#define PI 3.141592

__kernel void set_up_agents(__global float (*dogs_positions)[3],
                            __global float (*sheep_positions)[2],
                            __global float (*target_position),
                            __global int   (*seed))
{
    int id = get_global_id(0);
    int seed_value = seed[id];
    float x_pos = 10 + rand(&seed_value, AGENTS_LAYOUT_RANGE);
    float y_pos = 10 + rand(&seed_value, AGENTS_LAYOUT_RANGE);

    if (id < DOGS_COUNT)
    {
        float rotation = rand(&seed_value, 2 * PI);
        dogs_positions[id][0] = x_pos;
        dogs_positions[id][1] = y_pos;
        dogs_positions[id][2] = rotation;
    }
    else if (id - DOGS_COUNT < SHEEP_COUNT)
    {
        id -= DOGS_COUNT;
        sheep_positions[id][0] = x_pos;
        sheep_positions[id][1] = y_pos;
    }
    else
    {
        target_position[0] = x_pos;
        target_position[1] = y_pos;
    }
    seed[id] = seed_value;
}
