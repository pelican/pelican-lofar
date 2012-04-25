#ifndef GPU_NVIDIACONFIGURATION_H
#define GPU_NVIDIACONFIGURATION_H
#include <QList>
#include <QHash>
#include <QSet>
#include "GPU_MemoryMap.h"


/**
 * @file GPU_NVidiaConfiguration.h
 */

namespace pelican {

namespace lofar {
class GPU_Param;

/**
 * @class GPU_NVidiaConfiguration
 *  
 * @brief
 *    Describes the memory requirements for an NVidia card
 *    and provides an on demand resource mapping function
 * @details
 *    From a clean state - after a freeResources() call
 *    any devicePtr requests will be satisified - up to 
 *    the constraints of the card.
 *    Requests after a reset() will attempt to map
 *    devicePtr() requests to the existing memory allocation.
 *    If it is unable to do so, the request will throw 
 *    GPUConfigError.
 *
 */

class GPU_NVidiaConfiguration
{
    public:
        GPU_NVidiaConfiguration(  );
        ~GPU_NVidiaConfiguration();
        void reset();
        void freeResources();
        void syncOutputs();

        void* devicePtr( const GPU_MemoryMap& map );
        void* devicePtr( const GPU_MemoryMapOutput& map );
        void* devicePtr( const GPU_MemoryMapInputOutput& map );
        void* devicePtr( const GPU_MemoryMapConst& map );

    private:
        GPU_Param* _getParam( const GPU_MemoryMap& map );
        void _freeMem( const QList<GPU_Param*>& );

    private:
        int _paramIndex;
        QSet<GPU_Param*> _outputs;
        QHash<GPU_MemoryMap, GPU_Param* > _constantParams;
        QList<GPU_Param*> _params;
};

// Exception thrown when an incompatible config is made
class GPUConfigError 
{
    public:
        GPUConfigError() {};
        ~GPUConfigError() {};
};

} // namespace lofar
} // namespace pelican
#endif // GPU_NVIDIACONFIGURATION_H 
