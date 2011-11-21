#include "GPU_Manager.h"
#include "GPU_Resource.h"
#include "GPU_Job.h"
#include "GPU_NVidia.h"

namespace pelican {
namespace lofar {

/**
 *@details GPU_Manager 
 */
GPU_Manager::GPU_Manager(QObject* parent)
    : QThread(parent)
{
}

/**
 *@details
 */
GPU_Manager::~GPU_Manager()
{
     // clean up the resources
     foreach(GPU_Resource* r, _resources) {
         delete r;
     }
}

void GPU_Manager::addResource(GPU_Resource* r) {
     _resources.append(r);
     connect(r, SIGNAL(ready()), this, SLOT( _resourceFree() ) );
     _freeResource.append(r);
     _matchResources();
}

void GPU_Manager::run() {
    // Thread is staring up here
    // It is important that these objects are instantiated
    // in this thread.
    // Instantiate the cards e.g.
    // addResource(new GPU_NVidia);
    // import NVidia cards
#ifdef CUDA_FOUND
    GPU_NVidia::initialiseResources(this);
#endif

    // Start processing
    exec();

    // remove resources
     foreach(GPU_Resource* r, _resources) {
         delete r;
     }
     _resources.clear();
     _freeResource.clear();
}

void GPU_Manager::_matchResources() {
     if( _queue.size() > 0 ) {
        if( _freeResource.size() > 0 ) {
            _freeResource.takeFirst()->exec( _queue.takeFirst() );
        }
     }
}

void GPU_Manager::submit( const GPU_Job& job) {
     _queue.append(job);
     _matchResources();
} 

void GPU_Manager::_resourceFree() {
     _freeResource.append((GPU_Resource*)sender());
     _matchResources();
}

} // namespace lofar
} // namespace pelican
