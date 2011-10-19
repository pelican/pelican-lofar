#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H


/**
 * @file GPU_Kernel.h
 */

namespace pelican {

namespace lofar {

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
        virtual void run() = 0;

    private:
};

} // namespace lofar
} // namespace pelican
#endif // GPU_KERNEL_H 
