
__global__ void move_agents(Arrays *arrays)
{
    if (threadIdx.x < DOGS_COUNT)
    {
        move_dogs(arrays);
    }
    __syncthreads();

    move_sheep_$sheep_type(arrays);
}