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
    _currentConfig = new GPU_NVidiaConfiguration;
    cudaGetDeviceProperties(&_deviceProp, id);
}

/**
 *@details
 */
GPU_NVidia::~GPU_NVidia()
{
    delete _currentConfig;
    //cutilDeviceReset();
}

void GPU_NVidia::run( GPU_Job* job )
{
    // set to this device
    cudaSetDevice( _deviceId );
    // execute the kernels
    foreach( GPU_Kernel* kernel, job->kernels() ) {
       try { // try and reuse the existing configuration
           _currentConfig->reset();
           kernel->run( *this );
       }
       catch( const GPUConfigError& ) {
           // existing configuration is not compatible so
           // make a new one
           _currentConfig->freeResources();
           kernel->run( *this );
       }
       cudaDeviceSynchronize();
       if( ! cudaPeekAtLastError() ) {
           _currentConfig->syncOutputs();
       }
       else {
           throw( cudaGetErrorString( cudaPeekAtLastError() ) );
       }
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
