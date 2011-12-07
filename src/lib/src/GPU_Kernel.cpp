#include "GPU_Kernel.h"


namespace pelican {

namespace lofar {


/**
 *@details GPU_Kernel 
 */
GPU_Kernel::GPU_Kernel()
    : _config(0)
{
}

/**
 *@details
 */
GPU_Kernel::~GPU_Kernel()
{
}

void GPU_Kernel::setConfiguration( const GPU_NVidiaConfiguration* config )
{
    _config = config;
}

} // namespace lofar
} // namespace pelican
