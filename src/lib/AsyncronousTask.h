#ifndef ASYNCRONOUSTASK_H
#define ASYNCRONOUSTASK_H

#include <QObject>
#include <QHash>
#include <QMutex>
#include <QWaitCondition>
#include "pelican/data/DataBlob.h"
#include <boost/function.hpp>

/**
 * @file AsyncronousTask.h
 */

namespace pelican {

namespace lofar {

/**
 * @class AsyncronousTask
 *
 * @brief
 *   Base class for all Asyncronous Tasks
 *   This acts as the template for creating Asyncronous Jobs
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
                DataBlobFunctorMonitor() {};
                void operator()(DataBlob* blob) const;
        };
        friend class DataBlobFunctorMonitor;

    public:
        AsyncronousTask( const boost::function1<DataBlob*, DataBlob*>& workload );
        virtual ~AsyncronousTask();

        /// process the task in the background
        //  Will call run but in a separate thread and will return imediately
        void submit( DataBlob* data );

        /// A standardised run method. This will be executed in the current thread
        //  and will block until completion
        void run(DataBlob* data);

        /// schedule the provided task to be executed upon completion
        /// of this task
        //void link( AsyncronousTask* task );
        void link( const boost::function1<DataBlob*, DataBlob*>& functor );
        void onChainCompletion( const boost::function1<void, DataBlob*>& functor );

    signals:
        void finished( DataBlob* inputData );

    private:
        // called via functors
        void jobFinished( DataBlob* inputData, DataBlob* outputData );
        void taskFinished( DataBlob* );
        void subTaskFinished( AsyncronousTask*, DataBlob* );

    protected:
        // called to clean up and launch the necessary callbacks
        // when the task and all its dependents complete
        void _finished( DataBlob* inputData );

        // launchs a subtask. All subtasks must return before links are called.
        void submit( AsyncronousTask*, DataBlob* );

    private:
        QHash<DataBlob*, int> _dataLocker; // keep a track of subprocessing using
                                           // a specific DataBlob
        DataBlobFunctorMonitor _task;
        QList<DataBlobFunctorMonitor> _linkedFunctors;
        QList<CallBackT> _callBacks;
        QWaitCondition _subTaskWaitCondition;
        QMutex _subTaskMutex;
        int _subTaskCount;
};

} // namespace lofar
} // namespace pelican
#endif // ASYNCRONOUSTASK_H 
