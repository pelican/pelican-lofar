#include <QDebug>
#include "GPU_NVidia.h"
#include "GPU_Manager.h"
#include "GPU_Param.h"
#include "GPU_Kernel.h"
#include "GPU_Job.h"
#include "GPU_MemoryMap.h"
#include "GPU_NVidiaConfiguration.h"
#include <iostream>
#include <vector>
#include "stdio.h"

namespace pelican {

namespace lofar {

/**
 *@details GPU_NVidia 
 */
GPU_NVidia::GPU_NVidia( unsigned int id )
     :  _deviceId(id)
{
    cudaGetDeviceProperties(&_deviceProp, id);
}

/**
 *@details
 */
GPU_NVidia::~GPU_NVidia()
{
    freeMem( _params.values() );
    //cutilDeviceReset();
}

void GPU_NVidia::run( GPU_Job* job )
{
    // set to this device
    cudaSetDevice( _deviceId );
    // execute the kernels
    foreach( GPU_Kernel* kernel, job->kernels() ) {
       freeMem(_currentParams ); // quickfix: delete everything for now
       _currentParams.clear();
       _outputs.clear();
       kernel->run( *this );
       cudaDeviceSynchronize();
       if( ! cudaPeekAtLastError() ) {
           // copy device memory to host
           foreach( GPU_Param* param, _outputs ) {
                param->syncDeviceToHost();
           }
       }
       else {
           throw( cudaGetErrorString( cudaPeekAtLastError() ) );
        }
    }
}

void GPU_NVidia::freeMem( const QList<GPU_Param*>& list ) {
     foreach( GPU_Param* p, list ) {
        delete p;
        _params.remove(_params.key(p));
        _outputs.remove( p );
     }
}

void GPU_NVidia::initialiseResources(GPU_Manager* manager) {
     int num_devices=0;
     cudaGetDeviceCount(&num_devices);
     for(int i = 0; i < num_devices; i++) {
        manager->addResource( new GPU_NVidia( i ) );
     }
}

GPU_Param* GPU_NVidia::_getParam( const GPU_MemoryMap& map ) {
     if( ! _params.contains(map) ) {
         GPU_Param* p = new GPU_Param( map ) ;
         if(  cudaPeekAtLastError() ) {
             throw( cudaGetErrorString( cudaPeekAtLastError() ) );
         }
         _params.insert( map, p );
         _currentParams.append( p );
     }
     return _params.value(map);
}

void* GPU_NVidia::devicePtr( const GPU_MemoryMapOutput& map ) {
     GPU_Param* p = _getParam(map);
     _outputs.insert(p);
     return p->device();
}

void* GPU_NVidia::devicePtr( const GPU_MemoryMapInputOutput& map ) {
     GPU_Param* p = _getParam(map);
     _outputs.insert(p);
     p->syncHostToDevice();
     return p->device();
}

void* GPU_NVidia::devicePtr( const GPU_MemoryMap& map ) {
     GPU_Param* p = _getParam(map);
     p->syncHostToDevice();
     return p->device();
}

void* GPU_NVidia::devicePtr( const GPU_MemoryMapConst& map ) {
     if( ! _params.contains(map) ) {
         GPU_Param* p = new GPU_Param( map ) ;
         if(  cudaPeekAtLastError() ) {
             throw( cudaGetErrorString( cudaPeekAtLastError() ) );
         }
         _params.insert( map, p );
         _params.value(map)->syncHostToDevice(); // consts sync only on creation
         _currentParams.append( p );
     }
     return _params.value(map)->device();
}

} // namespace lofar
} // namespace pelican
