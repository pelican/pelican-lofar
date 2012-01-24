#include "GPU_Param.h"
#include "GPU_MemoryMap.h"
#include <cuda.h>
#include <cuda_runtime_api.h>


namespace pelican {

namespace lofar {


/**
 *@details GPU_Param 
 */
GPU_Param::GPU_Param( const GPU_MemoryMap* map ) : _map(map)
{
    cudaMalloc( &_devicePtr , map->size() );
}

/**
 *@details
 */
GPU_Param::~GPU_Param()
{
    cudaFree( _devicePtr );
}

void GPU_Param::syncHostToDevice() {
    if( _map->hostPtr() ) {
        cudaMemcpy( _devicePtr , _map->hostPtr(),
                _map->size(), cudaMemcpyHostToDevice );
    }
}

void GPU_Param::syncDeviceToHost() {
    if( _map->hostPtr() ) {
        cudaMemcpy( _map->hostPtr(), _devicePtr,
                _map->size(), cudaMemcpyDeviceToHost );
    }
}

unsigned long GPU_Param::size() const { 
    return _map->size(); 
}

void* GPU_Param::host() const { 
    return _map->hostPtr();
}

} // namespace lofar
} // namespace pelican
