
__kernel void get_furthest_sheep_distance(Arrays *arrays)
{
    int id = get_global_id(0);
    if (id == 0)
        distance = 0;
    barrier(CLK_GLOBAL_MEM_FENCE);
    float sheep_pos_x = arrays->sheep_positions[id][0];
    float sheep_pos_y = arrays->sheep_positions[id][1];
    float target_pos_x = arrays->target[0];
    float target_pos_y = arrays->target[1];

    int tmp_distance = (int)sqrtf(powf(sheep_pos_x - target_pos_x, 2) +
                               powf(sheep_pos_y - target_pos_y, 2));
    atomic_max(&distance, tmp_distance);
}
