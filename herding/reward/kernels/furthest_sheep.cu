
__device__ int distance;

__global__ void get_furthest_sheep_distance(Arrays *arrays)
{
    if (threadIdx.x == 0)
        distance = 0;
    __syncthreads();
    float sheep_pos_x = arrays->sheep_positions[threadIdx.x][0];
    float sheep_pos_y = arrays->sheep_positions[threadIdx.x][1];
    float target_pos_x = arrays->target[0];
    float target_pos_y = arrays->target[1];

    int tmp_distance = (int)sqrtf(powf(sheep_pos_x - target_pos_x, 2) +
                               powf(sheep_pos_y - target_pos_y, 2));
    atomicMax(&distance, tmp_distance);
}
