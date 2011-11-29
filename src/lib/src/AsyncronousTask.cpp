#include "AsyncronousTask.h"
#include <boost/bind.hpp>
#include <QtConcurrentRun>
#include <QMutexLocker>


namespace pelican {

namespace lofar {


/**
 *@details AsyncronousTask 
 */
AsyncronousTask::AsyncronousTask( const boost::function1<DataBlob*, DataBlob*>& task )
    : _task(task, this)
{
}

/**
 *@details
 */
AsyncronousTask::~AsyncronousTask()
{
}

void AsyncronousTask::jobFinished( DataBlob* inputData, DataBlob* outputData )
{
     if( _linkedFunctors.size() == 0 ) {
        // if there are no dependent tasks indicate we have finished
        _finished( inputData );
     }
     else {
        // pass the results of previous job down to any linked tasks
         foreach( const DataBlobFunctorMonitor& functor, _linkedFunctors ) { 
            ++_dataLocker[ inputData ];
            QtConcurrent::run( &functor, &DataBlobFunctorMonitor::operator(), outputData );
         }
     }
}

void AsyncronousTask::taskFinished( DataBlob* data ) {
     // count completion of all linked tasks
     if( --_dataLocker[data] == 0) {
        _dataLocker.remove(data);
        _finished(data);
     }
}

void AsyncronousTask::submit( DataBlob* data ) {
     QtConcurrent::run( this, &AsyncronousTask::run, data );
}

void AsyncronousTask::run( DataBlob* data ) {
    _task( data );
     // wait for any sub-tasks launched by task to finish 
     QMutexLocker lock(&_subTaskMutex);
     while( _subTaskCount )
        _subTaskWaitCondition.wait(&_subTaskMutex);
}

void AsyncronousTask::submit( AsyncronousTask* task, DataBlob* data ) {
     task->onChainCompletion( boost::bind(&AsyncronousTask::subTaskFinished, this, task, _1) );
     QMutexLocker lock(&_subTaskMutex);
     ++_subTaskCount;
     task->submit(data);
}

void AsyncronousTask::subTaskFinished( AsyncronousTask*, DataBlob* data ) {
     QMutexLocker lock(&_subTaskMutex);
     if( ! --_subTaskCount ) {
        _subTaskWaitCondition.wakeAll();
     }

}

/*
void AsyncronousTask::link( AsyncronousTask* task ) {
     link( boost::bind( &AsyncronousTask::run, task, _1) );
}
*/


void AsyncronousTask::link( const boost::function1<DataBlob*, DataBlob*>& functor ) {
     _linkedFunctors.append( DataBlobFunctorMonitor(functor, this) );
}

void AsyncronousTask::onChainCompletion( const boost::function1<void, DataBlob*>& functor ) {
    _callBacks.append(functor);
}

void AsyncronousTask::_finished(DataBlob* inputData)
{
     emit finished(inputData);
     // call each callback, sequentially
     foreach( const CallBackT& functor, _callBacks ) { 
        functor(inputData);
     }
}

AsyncronousTask::DataBlobFunctorMonitor::DataBlobFunctorMonitor (
                const DataBlobFunctorT& f, 
                AsyncronousTask* task 
       ) : _functor(f), _task(task)
{
     task->onChainCompletion( boost::bind( &AsyncronousTask::taskFinished, task, _1) );
}

void AsyncronousTask::DataBlobFunctorMonitor::operator()(DataBlob* inputData) const {
    _task->jobFinished( inputData, _functor(inputData) );
}

} // namespace lofar
} // namespace pelican
