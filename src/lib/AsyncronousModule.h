#ifndef ASYNCRONOUSMODULE_H
#define ASYNCRONOUSMODULE_H

#include <QMutex>
#include "pelican/core/AbstractModule.h"
#include <boost/function.hpp>
#include "ProcessingChain.h"

/**
 * @file AsyncronousModule.h
 */

namespace pelican {

namespace lofar {
class GPU_Job;
class GPU_Manager;

/**
 * @class AsyncronousModule
 *  
 * @brief
 *     Base class for Asyncronous Pelican Modules
 * @details
 * 
 */

class AsyncronousModule : public AbstractModule
{
    private:
        typedef boost::function1<void, DataBlob*> CallBackT;

    public:
        AsyncronousModule( const ConfigNode& config );
        virtual ~AsyncronousModule();
        void connect( const boost::function1<void, DataBlob*>& functor );
        void onChainCompletion( const boost::function0<void>& fn ) { _callbacks.append(fn); };

    protected:
        /// queue a GPU_Job for submission
        GPU_Job* submit(GPU_Job*);
        void exportData( DataBlob* data );
        void exportDataTrial( DataBlob* data );
        // mark DataBlob as being in use
        void lock( const DataBlob* );

        template<class DataBlobPtr>
        void lock( const QList<DataBlobPtr>& data ) {
            QMutexLocker lock(&_lockerMutex);
            foreach( const DataBlob* d, data ) {
                ++_dataLocker[d];
            }
        }
        // mark DataBlob as no longer being in use
        // returns the number of locks remaining ( 0 = unlocked )
        int unlock( const DataBlob* );
        void unlock( const QList<DataBlob*>& data );

    protected:
        ProcessingChain _chain;

    private:
        void _runTask( const CallBackT& functor, DataBlob* inputData );
        void _finished(); // call the chain completion callbacks
        QHash<const DataBlob*, int> _dataLocker; // keep a track of subprocessing using a specifc DataBlob
        QMutex _lockerMutex;
        static GPU_Manager* gpuManager();
        QList<CallBackT> _linkedFunctors;
        QList<boost::function0<void> > _callbacks; // end of chain callbacks

};

} // namespace lofar
} // namespace pelican
#endif // ASYNCRONOUSMODULE_H 
