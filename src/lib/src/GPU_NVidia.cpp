#include "GPU_NVidia.h"
#include "GPU_Manager.h"
#include "GPU_Job.h"
#include "GPU_MemoryMap.h"

namespace pelican {

namespace lofar {

/**
 *@details GPU_NVidia 
 */
GPU_NVidia::GPU_NVidia( unsigned int id )
{
    cudaGetDeviceProperties(&_deviceProp, id);
}

/**
 *@details
 */
GPU_NVidia::~GPU_NVidia()
{
}

void GPU_NVidia::run( const GPU_Job& job )
{
     // copy from host to device memory
     foreach( const boost::shared_ptr<GPU_MemoryMap>& map, job.inputMemoryMaps() ) {
         cudaMemcpy( map->start(), map->destination(), map->size(), cudaMemcpyHostToDevice );
     }

     // execute the kernels
     //foreach( GPU_Kernel& kernel, job.kernels() ) {
     //   kernel->run();
     //}

     // copy device to host
     foreach( const boost::shared_ptr<GPU_MemoryMap>& map, job.outputMemoryMaps() ) {
         cudaMemcpy( map->start() , map->destination(), map->size(), cudaMemcpyDeviceToHost );
     }
}

void GPU_NVidia::initialiseResources(GPU_Manager* manager) {
     int num_devices;
     cudaGetDeviceCount(&num_devices);
     for(int i = 0; i < num_devices; i++) {
        manager->addResource( new GPU_NVidia( i ) );
     }
}

} // namespace lofar
} // namespace pelican
