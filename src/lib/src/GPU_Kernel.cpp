#include "GPU_Kernel.h"


namespace pelican {

namespace lofar {


/**
 *@details GPU_Kernel 
 */
GPU_Kernel::GPU_Kernel()
{
}

/**
 *@details
 */
GPU_Kernel::~GPU_Kernel()
{
}

void GPU_Kernel::setConfiguration( const GPU_NVidiaConfiguration& config )
{
    _config = config;
}

void GPU_Kernel::clearInputOutputMaps() {
    _config.clearInputMaps();
    _config.clearOutputMaps();
}

} // namespace lofar
} // namespace pelican
