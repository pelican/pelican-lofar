#include "GPU_NVidia.h"


namespace pelican {

namespace lofar {


/**
 *@details GPU_NVidia 
 */
GPU_NVidia::GPU_NVidia()
{
}

/**
 *@details
 */
GPU_NVidia::~GPU_NVidia()
{
}

void GPU_NVidia::run( const GPU_Job& job)
{
     cudamemcpy( job.memStart(), job.memSize() );
     foreach( const GPU_Kernel& kernel, job.kernels() ) {
        kernel->run();
     }
}

} // namespace lofar
} // namespace pelican
