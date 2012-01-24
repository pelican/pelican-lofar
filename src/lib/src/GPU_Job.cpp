#include "GPU_Job.h"
#include "GPU_Kernel.h"


namespace pelican {

namespace lofar {


/**
 *@details GPU_Job 
 */
GPU_Job::GPU_Job()
    : _processing(false), _waitCondition(0)
{
    setStatus( GPU_Job::None );
}

// limited copy
// no status information
GPU_Job::GPU_Job( const GPU_Job& job )
    : _processing(false), _waitCondition(0)
{
     *this=job;
}

const GPU_Job& GPU_Job::operator=( const GPU_Job& job ) {
    _callbacks = job._callbacks;
    _kernels = job._kernels;
    setStatus( GPU_Job::None );
    return *this;
}

/**
 *@details
 */
GPU_Job::~GPU_Job()
{
    if( _waitCondition ) delete _waitCondition;
}

void GPU_Job::addKernel( GPU_Kernel* kernel )
{
    _kernels.append(kernel); 
}

void GPU_Job::reset() {
    _processing = false;
    setStatus( GPU_Job::None );
    _kernels.clear();
    _callbacks.clear();
}

void GPU_Job::wait() const {
    QMutexLocker lock(&_mutex);
    if( _processing  ) {
        _waitCondition=new QWaitCondition;
        _waitCondition->wait(&_mutex);
        delete _waitCondition;
        _waitCondition = 0;
    }
}

void GPU_Job::setAsRunning() {
    QMutexLocker lock(&_mutex);
    _processing = true;
}

void GPU_Job::emitFinished() {
    {
        QMutexLocker lock(&_mutex);
        _processing = false;
    }
    if( _waitCondition ) { 
        _waitCondition->wakeAll();
    }
}

} // namespace lofar
} // namespace pelican
