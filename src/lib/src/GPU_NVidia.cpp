#include "GPU_NVidia.h"
#include "GPU_Manager.h"
#include "GPU_Kernel.h"
#include "GPU_Job.h"
#include "GPU_MemoryMap.h"
#include "GPU_NVidiaConfiguration.h"
#include <iostream>
#include <vector>

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
    freeMem( _currentDevicePointers );
    //cutilDeviceReset();
}

void GPU_NVidia::run( GPU_Job* job )
{
     // set to this device
     cudaSetDevice( _deviceId );
     // execute the kernels
     foreach( GPU_Kernel* kernel, job->kernels() ) {
        setupConfiguration( kernel->configuration() );
        kernel->run( _currentDevicePointers );
        cudaDeviceSynchronize();
        if( ! cudaPeekAtLastError() ) {
            // copy device memory to host
            foreach( const GPU_MemoryMap& map, _currentConfig->outputMaps() ) {
                if( map.hostPtr() ) {
                    cudaMemcpy( map.hostPtr(), _memPointers[map],
                                map.size(), cudaMemcpyDeviceToHost );
                }
            }
        }
        else {
             throw( cudaGetErrorString( cudaPeekAtLastError() ) );
        }
     }
}

void GPU_NVidia::setupConfiguration ( const GPU_NVidiaConfiguration* c )
{
     if( _currentConfig != c ) {
         // free memory from existing job
         // TODO write code to test for overlapping mem
         // requirements for different configurations
         // to avoid unnesasary deallocations/allocations
         freeMem( _currentDevicePointers );
         _currentDevicePointers.clear();

         // allocate gpu memory
         foreach( const GPU_MemoryMap& map, c->allMaps() ) {
             if( ! _memPointers.contains(map) ) {
                 cudaMalloc( &(_memPointers[map]) , map.size() );
             }
             _currentDevicePointers.append( _memPointers[map] );
         }
         // deal with constant symbols - upload only once
         foreach( const GPU_MemoryMap& map, c->constants() ) {
             if( map.hostPtr() ) {
                 cudaMemcpy( _memPointers[map], map.hostPtr(),
                         map.size(), cudaMemcpyHostToDevice );
             }
         }
         _currentConfig = c;
     }
     // upload input data from host
     foreach( const GPU_MemoryMap& map, c->inputMaps() ) {
         if( map.hostPtr() ) {
             cudaMemcpy( _memPointers[map], map.hostPtr(),
                     map.size(), cudaMemcpyHostToDevice );
         }
     }
     cudaDeviceSynchronize();
}

void GPU_NVidia::freeMem( const QList<void*>& pointers ) {
     foreach( void* p, pointers ) {
        cudaFree( p );
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
