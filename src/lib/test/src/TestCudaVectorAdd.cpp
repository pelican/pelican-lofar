#include "TestCudaVectorAdd.h"

extern "C" void vecAdd( float*, float*, float* );

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

void TestCudaVectorAdd::run( const std::vector<void*>&  ) {
//     vecAdd( (float*)arg[0], (float*)arg[1], (float*)arg[2] );
}

} // namespace lofar
} // namespace pelican
