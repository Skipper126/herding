
float rand(unsigned int *seed, int range)
{
    *seed = ((*seed * 214013L + 2531011L) >> 16);
    return *seed % range;
}
