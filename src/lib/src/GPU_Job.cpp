#include "GPU_Job.h"
#include "GPU_Kernel.h"


namespace pelican {

namespace lofar {


/**
 *@details GPU_Job 
 */
GPU_Job::GPU_Job()
{
}

/**
 *@details
 */
GPU_Job::~GPU_Job()
{
}

void GPU_Job::addKernel( const GPU_Kernel& kernel )
{
    _kernels.append(&kernel); 
}


} // namespace lofar
} // namespace pelican
