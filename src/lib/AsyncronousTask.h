#ifndef ASYNCRONOUSTASK_H
#define ASYNCRONOUSTASK_H

#include <QObject>
#include <QHash>
#include <QFutureWatcher>
#include "pelican/data/DataBlob.h"
#include <boost/function.hpp>

/**
 * @file AsyncronousTask.h
 */

namespace pelican {

namespace lofar {
class AsyncronousJob;

/**
 * @class AsyncronousTask
 *
 * @brief
 *   Base class for all Asyncronous Tasks
 *   This acts as the template for creating AsyncronousJobs
 *   and provides monitoring of the whole chain of events
 * @details
 *   Note that there is an assumption that the Task will not be executed
 *   concurrently with the same DataBlob. Failure to adhere to this may cause 
 *   problems
 */

class AsyncronousTask : public QObject
{
    Q_OBJECT

    protected:
        typedef boost::function1<DataBlob*, DataBlob*> DataBlobFunctorT;
        typedef boost::function1<void, DataBlob*> CallBackT;
        class DataBlobFunctorMonitor {
              mutable DataBlobFunctorT _functor;
              AsyncronousTask* _task;
              public:
                DataBlobFunctorMonitor(const DataBlobFunctorT&, AsyncronousTask* task);
                void operator()(DataBlob* blob, AsyncronousJob* job ) const;
        };
        friend class DataBlobFunctorMonitor;
        friend class AsyncronousJob;

    public:
        AsyncronousTask(  );
        virtual ~AsyncronousTask();

        /// schedule the provided task to be executed upon completion
        /// of this task
        //void link( AsyncronousTask* task );
        void link( const boost::function1<DataBlob*, DataBlob*>& functor );

        void onChainCompletion( const boost::function1<void, DataBlob*>& functor );

    protected:
        ///  create a suitable asycronous job object to be associated
        //   with this task
        virtual AsyncronousJob* createJob( const DataBlobFunctorMonitor& , DataBlob* data ) = 0;

    signals:
        void finished( DataBlob* inputData );

    private:
        // called via functors
        void jobFinished( DataBlob* inputData, DataBlob* outputData, AsyncronousJob* job );
        void taskFinished( DataBlob* );

    protected:
        // called to clean up and launch the necessary callbacks
        // when the task and all its dependents complete
        void _finished( DataBlob* inputData );

    private:
        QHash<DataBlob*, int> _dataLocker; // keep a track of subprocessing using
                                           // a specific DataBlob
        QList<DataBlobFunctorMonitor> _linkedFunctors;
        QList<CallBackT> _callBacks;
        QList<AsyncronousJob*> _jobs;
};

} // namespace lofar
} // namespace pelican
#endif // ASYNCRONOUSTASK_H 
