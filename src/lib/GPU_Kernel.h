#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H
#include <QList>

/**
 * @file GPU_Kernel.h
 */

namespace pelican {

namespace lofar {
class GPU_NVidiaConfiguration;

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
        const GPU_NVidiaConfiguration* configuration() const { return _config; };
        virtual void run( const QList<void*>& devicePointers ) = 0;
        void setConfiguration( const GPU_NVidiaConfiguration* config );

    private:
        const GPU_NVidiaConfiguration* _config;
};

} // namespace lofar
} // namespace pelican
#endif // GPU_KERNEL_H 
