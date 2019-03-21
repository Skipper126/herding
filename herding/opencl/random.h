
float rand(int *seed, int range) // 1 <= *seed < m
{
    int const a = 16807; //ie 7**5
    int const m = 2147483647; //ie 2**31-1

    *seed = (long(*seed * a))%m;

    return ((*seed) / m ) * range;
}
