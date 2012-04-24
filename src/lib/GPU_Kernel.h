#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H
#include <QList>
#include "GPU_NVidiaConfiguration.h"
#include <GPU_Param.h>

/**
 * @file GPU_Kernel.h
 */

namespace pelican {

namespace lofar {
class GPU_NVidia;

/**
 * @class GPU_Kernel
 *  
 * @brief
 *    Base class for all Kernels
 * @details
 * 
 */

class GPU_Kernel
{
    public:
        GPU_Kernel(  );
        virtual ~GPU_Kernel();
        const GPU_NVidiaConfiguration* configuration() const { return &_config; };
        // implement this method to run the nvidia kernel
        // using GPU_MemoryMap type to transfer data
        virtual void run( GPU_NVidia& ) = 0;

    private:
        GPU_NVidiaConfiguration _config;
};

} // namespace lofar
} // namespace pelican
#endif // GPU_KERNEL_H 
