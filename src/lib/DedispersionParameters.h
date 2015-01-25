#ifndef DEDISPERSION_PARAMETERS_H_
#define DEDISPERSION_PARAMETERS_H_

// Shared memory
//#define NUMREG 10
//#define DIVINT 10
//#define DIVINDM 19

// L1 cache
#ifndef __CUDA_ARCH__
#warning "__CUDA_ARCH__ is not defined"
#define NUMREG 15
#define DIVINT 10
#define DIVINDM 20
#else
#if __CUDA_ARCH__ >= 200
#define NUMREG 12
#define DIVINT 6
#define DIVINDM 80
#define FDIVINDM 80.0f
#endif
#if (__CUDA_ARCH__ == 100 )
#define NUMREG 15
#define DIVINT 10
#define DIVINDM 20
#endif
#endif

#define NUMREG 12
#define DIVINT 6
#define DIVINDM 80
#define FDIVINDM 80.0f

#define ARRAYSIZE DIVINT * DIVINDM

#endif // DEDISPERSION_PARAMETERS_H_

