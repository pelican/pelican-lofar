#include "GPU_Manager.h"
#include "GPU_Resource.h"
#include "GPU_Job.h"


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
     // Thread staring up here
     // Instantiate the cards e.g.
     // addResource(new GPU_NVidia);

     // Start processing
     exec();
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
}; 

void GPU_Manager::_resourceFree() {
     _freeResource.append((GPU_Resource*)sender());
     _matchResources();
}


} // namespace lofar
} // namespace pelican
