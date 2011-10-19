#include "GPU_Resource.h"


namespace pelican {

namespace lofar {


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

void GPU_Resource::exec(const GPU_Job& job)
{
    run(job);
    emit ready();
}

} // namespace lofar
} // namespace pelican
