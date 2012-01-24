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

    protected:
        void setupConfiguration ( const GPU_NVidiaConfiguration* c );
        void freeMem( const QList<GPU_Param*>& );

    private:
        cudaDeviceProp _deviceProp;
        QHash<GPU_MemoryMap, GPU_Param* > _params;
        QList<GPU_Param*> _currentParams;
        const GPU_NVidiaConfiguration* _currentConfig;
        unsigned int _deviceId;
};

} // namespace lofar
} // namespace pelican
#endif // CUDA_FOUND
#endif // GPU_NVIDIA_H 
