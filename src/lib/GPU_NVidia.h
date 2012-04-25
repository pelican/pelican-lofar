#ifndef GPU_NVIDIA_H
#define GPU_NVIDIA_H

#ifdef CUDA_FOUND

#include "GPU_Resource.h"
#include "GPU_MemoryMap.h"
#include "GPU_Param.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <QHash>
#include <QList>
#include <QSet>

/**
 * @file GPU_NVidia.h
 */

namespace pelican {

namespace lofar {
class GPU_Manager;
class GPU_NVidiaConfiguration;

/**
 * @class GPU_NVidia
 *  
 * @brief
 *     An NVidia GPU Card wrapper for runnning GPU_Jobs via a GPU_Manager
 * @details
 * 
 */

class GPU_NVidia : public GPU_Resource
{

    public:
        GPU_NVidia( unsigned int id );
        ~GPU_NVidia();
        virtual void run( GPU_Job* job );

        static void initialiseResources(GPU_Manager* manager);

        void* devicePtr( const GPU_MemoryMap& map );
        void* devicePtr( const GPU_MemoryMapOutput& map );
        void* devicePtr( const GPU_MemoryMapInputOutput& map );
        void* devicePtr( const GPU_MemoryMapConst& map );

    protected:
        void freeMem( const QList<GPU_Param*>& );

    private:
        GPU_Param* _getParam( const GPU_MemoryMap& map );
        cudaDeviceProp _deviceProp;
        QHash<GPU_MemoryMap, GPU_Param* > _params;
        QList<GPU_Param*> _currentParams;
        QSet<GPU_Param*> _outputs;
        unsigned int _deviceId;
        GPU_NVidiaConfiguration* _currentConfig;
};

} // namespace lofar
} // namespace pelican
#endif // CUDA_FOUND
#endif // GPU_NVIDIA_H 
