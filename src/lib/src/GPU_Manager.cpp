#include "GPU_Manager.h"
#include <QMutexLocker>
#include <QtConcurrentRun>
#include <boost/bind.hpp>
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
     connect(r, SIGNAL(finished()), this, SLOT( _resourceFree() ) );
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
    // We will take all the available NVidia cards for now
    // This should really be made configurable in the future
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
        QMutexLocker lock(&_resourceMutex);
        if( _freeResource.size() > 0 ) {
            QtConcurrent::run( boost::bind( &GPU_Resource::exec, _freeResource.takeFirst(),  _queue.takeFirst())  );
        }
     }
}

void GPU_Manager::submit( GPU_Job* job) {
     job->setAsRunning(); // mark job as being dealt with
     _queue.append(job);
     _matchResources();
} 

void GPU_Manager::_resourceFree() {
     {
        QMutexLocker lock(&_resourceMutex);
        _freeResource.append((GPU_Resource*)sender());
     }
     _matchResources();
}

} // namespace lofar
} // namespace pelican
