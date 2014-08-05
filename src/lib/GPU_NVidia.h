#ifndef GPU_NVIDIA_H
#define GPU_NVIDIA_H

#ifdef CUDA_FOUND

#include "GPU_Resource.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "GPU_NVidiaConfiguration.h"

/**
 * @file GPU_NVidia.h
 */

namespace pelican {

namespace ampp {
class GPU_Manager;

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

        // call these functions from the kernel run() method to get GPU
        // memory resources.
        template<class MemMap> 
            void* devicePtr( const MemMap& map ) { return _currentConfig->devicePtr( map ); }

    private:
        cudaDeviceProp _deviceProp;
        unsigned int _deviceId;
        GPU_NVidiaConfiguration* _currentConfig;
};

} // namespace ampp
} // namespace pelican
#endif // CUDA_FOUND
#endif // GPU_NVIDIA_H 
