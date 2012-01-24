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
        virtual void run( const QList<GPU_Param*>& devicePointers ) = 0;
        void setConfiguration( const GPU_NVidiaConfiguration& config );
        inline void addConstant( const GPU_MemoryMap& map ) { _config.addConstant(map); };
        inline void addInputMap( const GPU_MemoryMap& map ) { _config.addInputMap(map); };
        inline void addOutputMap( const GPU_MemoryMap& map ) { _config.addOutputMap(map); };

    protected:
        GPU_NVidiaConfiguration _config;
};

} // namespace lofar
} // namespace pelican
#endif // GPU_KERNEL_H 
