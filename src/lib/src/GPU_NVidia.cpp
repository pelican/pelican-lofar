#include "GPU_NVidia.h"
#include "GPU_Manager.h"
#include "GPU_Kernel.h"
#include "GPU_Job.h"
#include "GPU_MemoryMap.h"
#include <iostream>
#include <vector>

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
    //cutilDeviceReset();
}

void GPU_NVidia::run( GPU_Job* job )
{
     // copy from host to device memory
     std::vector<void*> devicePointers;
     //foreach( const boost::shared_ptr<GPU_MemoryMap>& map, job->inputMemoryMaps() ) {
         // Allocate memory if required
         //if( ! _memPointers.contains(map.get()) ) {
         //    cudaMalloc( &_memPointer[map] , map->size() );
         //}
         //cudaMemcpy( map->start(), _memPointers[map], map->size(), cudaMemcpyHostToDevice );
     //}

     // execute the kernels
     foreach( GPU_Kernel* kernel, job->kernels() ) {
        kernel->run( devicePointers );
     }

     // copy device to host
     //foreach( const boost::shared_ptr<GPU_MemoryMap>& map, job->outputMemoryMaps() ) {
         //cudaMemcpy( map->hostPtr() , map->destination(), map->size(), cudaMemcpyDeviceToHost );
     //}
}

void GPU_NVidia::initialiseResources(GPU_Manager* manager) {
     int num_devices=0;
     cudaGetDeviceCount(&num_devices);
std::cout << "nVidia cards found: " << num_devices << std::endl;
     for(int i = 0; i < num_devices; i++) {
        manager->addResource( new GPU_NVidia( i ) );
     }
}

} // namespace lofar
} // namespace pelican
