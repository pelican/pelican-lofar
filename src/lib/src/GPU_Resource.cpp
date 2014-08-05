#include "GPU_Resource.h"
#include "GPU_Job.h"


namespace pelican {

namespace ampp {


/**
 *@details GPU_Resource 
 */
GPU_Resource::GPU_Resource()
{
}

/**
 *@details
 */
GPU_Resource::~GPU_Resource()
{
}

void GPU_Resource::exec( GPU_Job* job )
{
    run(job);
}

} // namespace ampp
} // namespace pelican
