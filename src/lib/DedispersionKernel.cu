#ifndef DEDISPERSE_KERNEL_H_
#define DEDISPERSE_KERNEL_H_


// Shared memory 
//#define NUMREG 10
//#define DIVINT 10
//#define DIVINDM 19

// L1 cache
#define NUMREG 15
#define DIVINT 15
//#define DIVINDM 57
#define DIVINDM 2
#define ARRAYSIZE DIVINT * DIVINDM

#include <iostream>

// Stores temporary shift values
//__device__ __constant__ float dm_shifts[8192];
//__device__ __constant__ int   i_nsamp, i_maxshift, i_nchans;
__device__ __shared__ float f_line[ARRAYSIZE];


//{{{ global_for_time_dedisperse_loop
__global__ void cache_dedisperse_loop(float *outbuff, float *buff, float mstartdm,
                                      float mdmstep, const float* dm_shifts,
                                      const int i_nsamp, const int* i_maxshift,
                                      const int* i_nchans )
{

    // NOTE: inshift AND outshift are set to 0 (zero) in the kernel call and so is
    // removed from this kernel.
    
    int   shift;    
    float local_kernel_t[NUMREG];

    int t  = blockIdx.x * NUMREG * DIVINT  + threadIdx.x;
    
    // Initialise the time accumulators
    for(int i = 0; i < NUMREG; i++) local_kernel_t[i] = 0.0f;

    float shift_temp = mstartdm + ((blockIdx.y * DIVINDM + threadIdx.y) * mdmstep);
    
    // Loop over the frequency channels.
    for(int c = 0; c < *i_nchans; c++) {


        // Calculate the initial shift for this given frequency
        // channel (c) at the current despersion measure (dm) 
        // ** dm is constant for this thread!!**
        shift = (c * i_nsamp + t) + __float2int_rz (dm_shifts[c] * shift_temp);
        
        #pragma unroll
        for(int i = 0; i < NUMREG; i++) {
            local_kernel_t[i] += buff[shift + (i * DIVINT) ];
        }
    }

    // Write the accumulators to the output array. 
    #pragma unroll
    for(int i = 0; i < NUMREG; i++) {
        outbuff[((blockIdx.y * DIVINDM) + threadIdx.y)* (i_nsamp-*i_maxshift) + (i * DIVINT) + (NUMREG * DIVINT * blockIdx.x) + threadIdx.x] = local_kernel_t[i];
       //outbuff[((blockIdx.y * DIVINDM) + threadIdx.y)* (i_nsamp) + (i * DIVINT) + (NUMREG * DIVINT * blockIdx.x) + threadIdx.x] = local_kernel_t[i];
    }
}

/// C Wrapper for brute-force algo
extern "C" void cacheDedisperseLoop( float *outbuff, long outbufSize, float *buff, float mstartdm,
                                     float mdmstep, int tdms, int numSamples, 
                                     const float* dmShift,
                                     const int* i_maxshift, 
                                     const int* i_nchans ) {

    cudaMemset(outbuff, 0, outbufSize );
    int divisions_in_t  = DIVINT;
    int divisions_in_dm = DIVINDM;
    int num_reg = NUMREG;
    int num_blocks_t = numSamples/(divisions_in_t * num_reg) || 1 ;
    int num_blocks_dm = tdms/divisions_in_dm;


    std::cout << "\nnumSamples\t" << numSamples;
    std::cout << "\ndivisions_in_t\t" << divisions_in_t;
    std::cout << "\ndivisions_in_dm\t" << divisions_in_dm;
    std::cout << "\nnum_reg\t" << num_reg;
    std::cout << "\nnum_blocks\t" << num_blocks_t;
    std::cout << "\nnum_blocks_dm\t" << num_blocks_dm;
    //printf("\ndm_step\t%f", dm_step);
    std::cout << "\ntdms\t" << tdms << std::endl;
    //std::cout << "\nmaxshift\t" << *i_maxshift << std::endl;
    //std::cout << "\nnchans\t" << *i_nchans << std::endl;
    //std::cout << "\ndmshift\t" << *dmShift << std::endl;
    std::cout << "mdmstep\t" << mdmstep << std::endl;
    std::cout << "mstartdm\t" << mstartdm << std::endl;
    std::cout << "buff\t" << buff << std::endl;
    std::cout << "outbuff\t" << outbuff << std::endl;

    dim3 threads_per_block(divisions_in_t, divisions_in_dm);
    dim3 num_blocks(num_blocks_t,num_blocks_dm);

    cache_dedisperse_loop<<< num_blocks, threads_per_block >>>( outbuff, buff, 
                mstartdm, mdmstep, dmShift, numSamples, i_maxshift, i_nchans );
}

//}}}

//{{{ shared_global_time_dedisperse_loop

/*
__global__ void shared_dedisperse_loop(float *outbuff, float *buff, float mstartdm, float mdmstep)
{
    int   i, c, shift;    
    float local_kernel_t[NUMREG];

    // Initialise the time accumulators
    for(i = 0; i < NUMREG; i++) local_kernel_t[i] = 0.0f;

    int shift_one = (mstartdm +((blockIdx.y*DIVINDM + threadIdx.y)*mdmstep));
    int shift_two = (mstartdm + (blockIdx.y*DIVINDM*mdmstep));
    int idx = (threadIdx.x + (threadIdx.y * DIVINT));
        for(c = 0; c < i_nchans; c++) {
        
        f_line[idx] = buff[((c*i_nsamp) + (blockIdx.x*NUMREG*DIVINT + threadIdx.x)) + __float2int_rz(dm_shifts[c]*shift_one)];
        __syncthreads();
        
        shift = __float2int_rz(dm_shifts[c]*shift_one) - __float2int_rz(dm_shifts[c]*shift_two);
        for(i = 0; i < NUMREG; i++) {
            local_kernel_t[i] += f_line[(shift + (i*DIVINT))];
        }
    }

    // Write the accumulators to the output array. 
    shift = ((blockIdx.y*DIVINDM) + threadIdx.y)*(i_nsamp-i_maxshift) + (NUMREG*DIVINT*blockIdx.x) + threadIdx.x;
    for(i = 0; i < NUMREG; i++) {
        outbuff[shift + (i*DIVINT)] = local_kernel_t[i];
    }
}
*/

//}}}

#endif
