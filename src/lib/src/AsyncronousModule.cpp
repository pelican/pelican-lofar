#include "AsyncronousModule.h"
#include "GPU_Manager.h"
#include "GPU_NVidia.h"


namespace pelican {

namespace lofar {


/**
 *@details AsyncronousModule 
 */
AsyncronousModule::AsyncronousModule( const ConfigNode& config )
    : AbstractModule( config )
{
   // initialise the GPU manager if required
   // for now we share the mamanger between all instances
   // and hog all the cards. We could refine this by removing
   // the static and passing
   // down an appropriately configured gpuManager in the 
   // constructor.
   if( gpuManager()->resources() == 0 ) {
       GPU_NVidia::initialiseResources( gpuManager() );
   }
}

GPU_Manager* AsyncronousModule::gpuManager() {
    static GPU_Manager gpuManager;
    return &gpuManager;
}

/**
 *@details
 */
AsyncronousModule::~AsyncronousModule()
{
}

GPU_Job* AsyncronousModule::submit(GPU_Job* job) {
    return gpuManager()->submit(job);
}

} // namespace lofar
} // namespace pelican
