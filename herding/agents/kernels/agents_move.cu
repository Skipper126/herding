
__global__ void move_agents(Arrays *arrays)
{
    if (threadIdx.x < DOGS_COUNT)
    {
        move_dogs(arrays);
    }

    __syncthreads();

    if (threadIdx.x < SHEEP_COUNT)
    {
        move_sheep_$sheep_type(arrays);
    }
}
