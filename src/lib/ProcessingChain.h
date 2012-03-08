#ifndef PROCESSINGCHAIN_H
#define PROCESSINGCHAIN_H
#include <QMutex>
#include <QHash>
#include <boost/function.hpp> 


/**
 * @file ProcessingChain.h
 */

namespace pelican {

namespace lofar {

/**
 * @class ProcessingChain
 *  
 * @brief
 *    A container to launch and monitor processing stages
 * @details
 * 
 */

class ProcessingChain
{
    private:
        typedef boost::function0<void> CallBackT;

    public:
        ProcessingChain();
        ~ProcessingChain();
        /// execute the chain, starting with the parallel tasks
        //  and then the post completion task (sequential)
        //  This function is thread safe and re-entrant
        void exec( const QList<CallBackT>& parallelTasks, const QList<CallBackT>& postTasks);

    private:
        void _runTask( const CallBackT& functor, unsigned taskId, const QList<CallBackT>& postTasks );
        void _finished( const QList<CallBackT>& postProcessingTasks ); // call completion callbacks
        QMutex _mutex;
        QHash<unsigned, unsigned> _processCount; // keep a track of threads per _taskId
        unsigned _taskId; // unique identifier for each call to exec()
};

} // namespace lofar
} // namespace pelican
#endif // PROCESSINGCHAIN_H 
