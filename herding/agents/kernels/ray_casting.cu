

__global__ void cast_rays(Arrays *arrays)
{
    arrays.observation[threadIdx.x] = threadIdx.x;

}