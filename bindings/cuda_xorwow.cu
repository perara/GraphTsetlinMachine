#include <curand_kernel.h>

extern "C" {
__global__ void prepare_xorwow(curandStateXORWOW *state, const size_t n,
                               unsigned int *seeds, const size_t offset)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        curand_init(seeds[id], id, offset, &state[id]);
}

__global__ void skip_ahead_sequence_xorwow(curandStateXORWOW *state, const size_t n, const unsigned long long skip)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        skipahead_sequence(skip, &state[idx]);
}

__global__ void skip_ahead_sequence_array_xorwow(curandStateXORWOW *state, const size_t n, const unsigned long long *skip)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        skipahead_sequence(skip[idx], &state[idx]);
}

__global__ void generate_uniform(curandStateXORWOW *state, const size_t n, float *result)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        result[idx] = curand_uniform(&state[idx]);
}
}