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


void GPU_Job::setInputMap( const boost::shared_ptr<GPU_MemoryMap>& map ) {
    _inputMaps.append(map);
}

void GPU_Job::setOutputMap( const boost::shared_ptr<GPU_MemoryMap>& map ) {
    _outputMaps.append(map);
}

} // namespace lofar
} // namespace pelican
