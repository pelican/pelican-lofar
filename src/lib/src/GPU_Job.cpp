#include "GPU_Job.h"
#include "GPU_Kernel.h"


namespace pelican {

namespace lofar {


/**
 *@details GPU_Job 
 */
GPU_Job::GPU_Job()
{
}

/**
 *@details
 */
GPU_Job::~GPU_Job()
{
}

void GPU_Job::addKernel( const GPU_Kernel& kernel )
{
    _kernels.append(&kernel); 
}


void GPU_Job::setInputMap( const boost::shared_ptr<GPU_MemoryMap>& map ) {
    _inputMaps.append(map);
}

void GPU_Job::setOutputMap( const boost::shared_ptr<GPU_MemoryMap>& map ) {
    _outputMaps.append(map);
}

void GPU_Job::wait() const {
    QMutexLocker lock(&_mutex);
    while( _processing  ) 
        _waitCondition.wait(&_mutex);
}

void GPU_Job::setAsRunning() {
    QMutexLocker lock(&_mutex);
    _processing = true;
}

void GPU_Job::emitFinished() {
    QMutexLocker lock(&_mutex);
    _processing = false;
    Q_EMIT jobFinished();
}

} // namespace lofar
} // namespace pelican
