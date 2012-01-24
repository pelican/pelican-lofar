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

void TestCudaVectorAdd::run( const QList<GPU_Param*>& args  ) {
     vecAdd( (const float*)args[0]->device(), (const float*)args[1]->device(), (float*)args[2]->device(), 2 );
}

} // namespace lofar
} // namespace pelican
