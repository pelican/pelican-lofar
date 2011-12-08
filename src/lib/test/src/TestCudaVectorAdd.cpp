#include "TestCudaVectorAdd.h"

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

void TestCudaVectorAdd::run( const QList<void*>& args  ) {
     vecAdd( (const float*)args[0], (const float*)args[1], (float*)args[2], 2 );
}

} // namespace lofar
} // namespace pelican
