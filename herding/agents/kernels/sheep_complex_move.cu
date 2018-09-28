
__device__ void move_sheep_complex(Arrays *arrays)
{

    arrays->sheep_positions[threadIdx.x][0] += arrays->rand_values[threadIdx.x];
    arrays->sheep_positions[threadIdx.x][1] -= arrays->rand_values[threadIdx.x];
    move_sheep_simple(arrays);
}
