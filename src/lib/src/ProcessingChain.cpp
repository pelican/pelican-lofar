#include "ProcessingChain.h"
#include <QMutexLocker>
#include <QtConcurrentRun>

namespace pelican {

namespace lofar {


/**
 *@details ProcessingChain 
 */
ProcessingChain::ProcessingChain()
        : _taskId(0)
{
}

/**
 *@details
 */
ProcessingChain::~ProcessingChain()
{
    // wait for all processing tasks to finish
    waitTaskCompletion();
}

void ProcessingChain::waitTaskCompletion() const {
    while( _processCount.size() ) {
        usleep(10);
    }
}

void ProcessingChain::exec( const QList<CallBackT>& parallelTasks,
                            const QList<CallBackT>& postProcessingTasks ) {
     if( parallelTasks.size() == 0 ) {
        // if there are no dependent tasks indicate we have finished
        _finished( postProcessingTasks );
     }
     else {
        QMutexLocker lock(&_mutex);
        _processCount[++_taskId]=0;
        // each task is launched in a separate thread
        foreach( const CallBackT& functor, parallelTasks ) { 
            ++_processCount[_taskId];
            QtConcurrent::run( this, &ProcessingChain::_runTask, functor, _taskId, postProcessingTasks );
        }
     }
}

void ProcessingChain::_runTask( const CallBackT& functor, unsigned taskId, const QList<CallBackT>& postProcessingTasks ) {
     functor();
     QMutexLocker lock(&_mutex);
     if( --_processCount[taskId] == 0) {
        _finished( postProcessingTasks );
        _processCount.remove(taskId); // mark processing chain complete
     }
}

void ProcessingChain::_finished( const QList<CallBackT>& postProcessingTasks )
{
    // call each post chain event sequentialy
    foreach( const boost::function0<void>& fn, postProcessingTasks ) {
        fn();
    }
}

} // namespace lofar
} // namespace pelican
