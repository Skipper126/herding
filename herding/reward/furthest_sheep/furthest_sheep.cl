#include "herding/data/env_data.h"

__kernel void get_furthest_sheep_distance(__global struct Arrays *arrays)
{
    __local int distance;
    int id = get_global_id(0);
    if (id == 0)
        distance = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    float sheep_pos_x = arrays->sheep_positions[id][0];
    float sheep_pos_y = arrays->sheep_positions[id][1];
    float target_pos_x = arrays->target_position[0];
    float target_pos_y = arrays->target_position[1];

    int tmp_distance = (int)sqrt(pow(sheep_pos_x - target_pos_x, 2) +
                               pow(sheep_pos_y - target_pos_y, 2));
    atomic_max(&distance, tmp_distance);

    barrier(CLK_LOCAL_MEM_FENCE);
    if(id == 0)
        arrays->common_output = distance;
}
