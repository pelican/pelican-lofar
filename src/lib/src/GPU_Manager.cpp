#include "GPU_Manager.h"
#include <QMutexLocker>
#include <QtConcurrentRun>
#include <boost/bind.hpp>
#include "GPU_Resource.h"
#include "GPU_Job.h"
#include "GPU_NVidia.h"
#include <iostream>

namespace pelican {
namespace lofar {

/**
 *@details GPU_Manager 
 */
GPU_Manager::GPU_Manager()
{
    _destructor = false;
}

/**
 *@details
 */
GPU_Manager::~GPU_Manager()
{
    _destructor = true;
    QMutexLocker lock(&_resourceMutex);
    // clean up the resources
    foreach(GPU_Resource* r, _resources) {
        delete r;
    }
    _resources.clear();
    _freeResource.clear();
    _queue.clear();
}

void GPU_Manager::addResource(GPU_Resource* r) {
    QMutexLocker lock(&_resourceMutex);
    _resources.append(r);
    _freeResource.append(r);
    _matchResources();
}

int GPU_Manager::resources() const {
    return _resources.size();
}

void GPU_Manager::_matchResources() {
     // ensure _resourceMutex is locked before calling this 
     // function
     if( _queue.size() > 0 ) {
        if( _freeResource.size() > 0 ) {
            QtConcurrent::run( this, &GPU_Manager::_runJob, _freeResource.takeFirst(), _queue.takeFirst() );
        }
     }
}

void GPU_Manager::_runJob( GPU_Resource* r, GPU_Job* job ) {
    job->setStatus( GPU_Job::Running );
    r->exec(job);
    job->setStatus( GPU_Job::Finished );
    job->emitFinished();
    _resourceFree( r );
    // execute any job callbacks
    foreach( const boost::function0<void>& fn, job->callBacks() ) {
        fn();
    }
}

int GPU_Manager::freeResources() const {
    QMutexLocker lock(&_resourceMutex);
    return _freeResource.size();
}

int GPU_Manager::jobsQueued() const {
    QMutexLocker lock(&_resourceMutex);
    return _queue.size();
}

GPU_Job* GPU_Manager::submit( GPU_Job* job) {
    job->setStatus( GPU_Job::Queued );
    job->setAsRunning(); // mark job as being dealt with
    QMutexLocker lock(&_resourceMutex);
    _queue.append(job);
    _matchResources();
    return job;
} 

void GPU_Manager::_resourceFree( GPU_Resource* res) {
    if( ! _destructor ) {
        QMutexLocker lock(&_resourceMutex);
        _freeResource.append(res);
        _matchResources();
    }
}

} // namespace lofar
} // namespace pelican
