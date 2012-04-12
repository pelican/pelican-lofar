#include "GPU_Param.h"
#include "GPU_MemoryMap.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>


namespace pelican {

namespace lofar {


/**
 *@details GPU_Param 
 */
GPU_Param::GPU_Param( const GPU_MemoryMap& map ) : _map(map), _devicePtr(0)
{
    cudaMalloc( &_devicePtr , map.size() );
}

/**
 *@details
 */
GPU_Param::~GPU_Param()
{
    cudaFree( _devicePtr );
}

void GPU_Param::syncHostToDevice() {
    if( _map.hostPtr() ) {
//        std::cout << "GPU_Param::syncHostToDevice: device=" << _devicePtr << " host=" << _map.hostPtr() << " size=" << _map.size() << std::endl;
        cudaMemcpy( _devicePtr , _map.hostPtr(),
                _map.size(), cudaMemcpyHostToDevice );
    }
}

void GPU_Param::syncDeviceToHost() {
    if( _map.hostPtr() ) {
//        std::cout << "GPU_Param::syncDeviceToHost: device=" << _devicePtr << " host=" << _map.hostPtr() << " size=" << _map.size() << std::endl;
        cudaMemcpy( _map.hostPtr(), _devicePtr,
                _map.size(), cudaMemcpyDeviceToHost );
    }
}

unsigned long GPU_Param::size() const { 
    return _map.size(); 
}

void* GPU_Param::host() const { 
    return _map.hostPtr();
}

} // namespace lofar
} // namespace pelican
