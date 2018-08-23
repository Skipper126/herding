

__device__ void move_sheep_simple(Arrays *arrays)
{
    arrays.sheep_positions[threadIdx.x][0] += 1;
    arrays.sheep_positions[threadIdx.x][1] += 1;
}