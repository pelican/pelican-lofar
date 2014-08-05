#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H
#include <QList>
#include <GPU_Param.h>

/**
 * @file GPU_Kernel.h
 */

namespace pelican {

namespace ampp {
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
        // implement this method to run the nvidia kernel
        // using GPU_MemoryMap type to transfer data
        virtual void run( GPU_NVidia& ) = 0;
        // this method will be called when something
        // goes wrong and the run is abandoned.
        // call any callbacks for the MemoryMap from here
        virtual void cleanUp() {};

    private:
};

} // namespace ampp
} // namespace pelican
#endif // GPU_KERNEL_H 
