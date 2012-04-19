#ifndef PROCESSINGCHAIN1_H
#define PROCESSINGCHAIN1_H
#include <QMutex>
#include <QHash>
#include <boost/function.hpp> 
#include <QtConcurrentRun>


/**
 * @file ProcessingChain1.h
 */

namespace pelican {

namespace lofar {

/**
 * @class ProcessingChain1
 *  
 * @brief
 *    A container to launch and monitor processing stages
 *    Functors for the Parrallel stages to take a single argument
 * @details
 * 
 */

template<typename argT>
class ProcessingChain1
{
    private:
        typedef boost::function1<void, argT> CallBackT;
        typedef boost::function0<void> PostCallBackT;

    public:
        ProcessingChain1() : _taskId(0) {};
        ~ProcessingChain1() {
            waitTaskCompletion();
        };

        void waitTaskCompletion() const {
            while( _processCount.size() ) {
                usleep(10);
            }
        }

        /// execute the chain, starting with the parallel tasks
        //  and then the post completion task (sequential)
        //  This function is thread safe and re-entrant
        void exec( const QList<CallBackT>& parallelTasks, const QList<PostCallBackT>& postTasks, const argT& arg ) 
        {
                 if( parallelTasks.size() == 0 ) {
                    // if there are no dependent tasks indicate we have finished
                    _finished( postTasks );
                 }
                 else {
                    QMutexLocker lock(&_mutex);
                    _processCount[++_taskId]=0;
                    // each task is launched in a separate thread
                    foreach( const CallBackT& functor, parallelTasks ) { 
                        ++_processCount[_taskId];
                        QtConcurrent::run( this, &ProcessingChain1::_runTask, functor, 
                                           _taskId, postTasks, arg 
                                         );
                    }
                 }
        }

    private:
        void _runTask( const CallBackT& functor, unsigned taskId, 
                       const QList<PostCallBackT>& postTasks, const argT& arg )
        {
             functor( arg );
             QMutexLocker lock(&_mutex);
             if( --_processCount[taskId] == 0) {
                _finished( postTasks );
                _processCount.remove(taskId);
             }
        }

        void _finished( const QList<PostCallBackT>& postProcessingTasks )
        {
            // call completion callbacks
            foreach( const PostCallBackT& fn, postProcessingTasks ) {
                fn();
            }
        }

     private:
        QMutex _mutex;
        QHash<unsigned, unsigned> _processCount; // keep a track of threads per _taskId
        unsigned _taskId; // unique identifier for each call to exec()
};

} // namespace lofar
} // namespace pelican
#endif // PROCESSINGCHAIN1_H 
