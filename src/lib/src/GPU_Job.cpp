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


void GPU_Job::addInputMap( const GPU_MemoryMap& map ) {
    _inputMaps.append(map);
}

void GPU_Job::addOutputMap( const GPU_MemoryMap& map ) {
    _outputMaps.append(map);
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
        setStatus( GPU_Job::Finished );
        _processing = false;
    }
    if( _waitCondition ) { 
        _waitCondition->wakeAll();
    }
}

} // namespace lofar
} // namespace pelican
