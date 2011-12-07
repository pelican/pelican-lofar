// CUDA code : Add two float vectors together
// Device code ( taken form Cuda SDK )
#include <iostream>

__global__ void VecAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// C interface wrapper - A B C are dudaMalloc'ed references
extern "C" void vecAdd( const float* A, const float* B, float* C, int size ) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, size);
}

