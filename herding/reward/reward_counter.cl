
__kernel void get_medium_distance(__global float (*sheep_positions)[2],
                                    __global float *target_position,
                                    __global int *output)
{
    __local int distance;
    int id = get_global_id(0);
    if (id == 0)
        distance = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    float sheep_pos_x = sheep_positions[id][0];
    float sheep_pos_y = sheep_positions[id][1];
    float target_pos_x = target_position[0];
    float target_pos_y = target_position[1];

    int tmp_distance = (int)sqrt(pow(sheep_pos_x - target_pos_x, 2) +
                               pow(sheep_pos_y - target_pos_y, 2)) - HERD_TARGET_RADIUS;
    tmp_distance = tmp_distance < 0 ? 0 : tmp_distance;
    atomic_add(&distance, tmp_distance);

    barrier(CLK_LOCAL_MEM_FENCE);
    if(id == 0)
    {
        *output = distance / SHEEP_COUNT;
    }
}
