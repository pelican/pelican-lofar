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
     : _currentConfig(0), _deviceId(id)
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
       setupConfiguration( &(kernel->configuration()) );
       kernel->run( _currentParams );
       cudaDeviceSynchronize();
       if( ! cudaPeekAtLastError() ) {
           // copy device memory to host
           foreach( const GPU_MemoryMap& map, _currentConfig->outputMaps() ) {
                _params.value(map)->syncDeviceToHost();
           }
       }
       else {
           throw( cudaGetErrorString( cudaPeekAtLastError() ) );
        }
    }
}

void GPU_NVidia::setupConfiguration ( const GPU_NVidiaConfiguration* c )
{
     //if( _currentConfig != c ) { // ! assumes config does not change between invoca.
         // free memory from existing job
         // TODO write code to test for overlapping mem
         // requirements for different configurations
         // to avoid unnessasary deallocations/allocations
         freeMem(_currentParams ); // quickfix: delete everything for now
         _currentParams.clear();
         foreach( const GPU_MemoryMap& map, c->allMaps() ) {
             GPU_Param* p = new GPU_Param( map ) ;
             if(  cudaPeekAtLastError() ) {
                throw( cudaGetErrorString( cudaPeekAtLastError() ) );
             }
             _params.insert( map, p );
             _currentParams.append( p );
         }
         // sync constants only on creation
         foreach( const GPU_MemoryMap& map, c->constants() ) {
             _params.value(map)->syncHostToDevice();
             if(  cudaPeekAtLastError() ) {
                 throw( cudaGetErrorString( cudaPeekAtLastError() ) );
             }
         }
         _currentConfig = c;
     //}
     // upload non-constant input data from host
     foreach( const GPU_MemoryMap& map, c->inputMaps() ) {
         Q_ASSERT( _params.contains(map) );
         _params.value(map)->syncHostToDevice();
         if(  cudaPeekAtLastError() ) {
             throw( cudaGetErrorString( cudaPeekAtLastError() ) );
         }
     }
     cudaDeviceSynchronize();
}

void GPU_NVidia::freeMem( const QList<GPU_Param*>& list ) {
     foreach( GPU_Param* p, list ) {
        delete p;
        _params.remove(_params.key(p));
     }
}

void GPU_NVidia::initialiseResources(GPU_Manager* manager) {
     int num_devices=0;
     cudaGetDeviceCount(&num_devices);
     for(int i = 0; i < num_devices; i++) {
        manager->addResource( new GPU_NVidia( i ) );
     }
}

} // namespace lofar
} // namespace pelican
