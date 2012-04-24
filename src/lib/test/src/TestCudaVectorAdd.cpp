#include "TestCudaVectorAdd.h"
#include "GPU_NVidia.h"

extern "C" void vecAdd( const float*, const float*, float*, int );

namespace pelican {

namespace lofar {


/**
 *@details TestCudaVectorAdd 
 */
TestCudaVectorAdd::TestCudaVectorAdd()
    : GPU_Kernel()
{
}

/**
 *@details
 */
TestCudaVectorAdd::~TestCudaVectorAdd()
{
}

void TestCudaVectorAdd::run( GPU_NVidia& gpu ) {
     vecAdd( (const float*)gpu.devicePtr(_vec1), (const float*)gpu.devicePtr(_vec2)
              , (float*)gpu.devicePtr(_vecOut), 2 );
}

} // namespace lofar
} // namespace pelican
