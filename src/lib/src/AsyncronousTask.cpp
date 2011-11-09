#include "AsyncronousTask.h"
#include "AsyncronousJob.h"
#include <boost/bind.hpp>


namespace pelican {

namespace lofar {


/**
 *@details AsyncronousTask 
 */
AsyncronousTask::AsyncronousTask()
{
}

/**
 *@details
 */
AsyncronousTask::~AsyncronousTask()
{
     foreach( AsyncronousJob* job, _jobs ) {
        delete job;
     }
}

void AsyncronousTask::jobFinished( DataBlob* inputData, DataBlob* outputData, AsyncronousJob* job  )
{
     if( _linkedFunctors.size() == 0 ) {
        // if there are no dependent tasks indicate we have finished
        _finished( inputData );
     }
     else {
        // pass the results of previous job down to any linked tasks
         foreach( const DataBlobFunctorMonitor& functor, _linkedFunctors ) { 
            ++_dataLocker[ inputData ];
            AsyncronousJob* job=this->createJob( functor, outputData );
            _jobs.append(job);
            job->submit();
         }
     }
     _jobs.removeOne(job);
     delete job;
}

void AsyncronousTask::taskFinished( DataBlob* data ) {
     // count completion of all linked tasks
     if( --_dataLocker[data] == 0) {
        _dataLocker.remove(data);
        _finished(data);
     }
}

//void AsyncronousTask::link( AsyncronousTask* task,  ) {
//     link( boost::bind( &AsyncronousTask::submit, task, _1) );
//}

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

void AsyncronousTask::DataBlobFunctorMonitor::operator()(DataBlob* inputData, AsyncronousJob* job) const {
    _task->jobFinished( inputData, _functor(inputData), job );
}

} // namespace lofar
} // namespace pelican
