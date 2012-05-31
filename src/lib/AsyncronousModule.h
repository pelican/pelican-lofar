#ifndef ASYNCRONOUSMODULE_H
#define ASYNCRONOUSMODULE_H

#include <QMutex>
#include "pelican/core/AbstractModule.h"
#include <boost/function.hpp>
#include "ProcessingChain1.h"
#include <iostream>

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
    public:
        typedef boost::function1<void, DataBlob*> CallBackT;
        typedef boost::function1<void, const QList<DataBlob*>& > UnlockCallBackT;

    public:
        AsyncronousModule( const ConfigNode& config );
        virtual ~AsyncronousModule();
        /// attach a task to be processed when the data is available
        //  Each task will be run in a separate thread
        void connect( const boost::function1<void, DataBlob*>& functor );

        /// attach a task to be completed when a DataBlob is unlocked by the module
        void unlockCallback( const UnlockCallBackT& callback );

        /// attach a task to be processed when all connect tasks have been completed
        //  and all locks removed
        void onChainCompletion( const boost::function0<void>& fn ) { _callbacks.append(fn); };

        /// return the number of locks for the specified object
        int lockNumber( const DataBlob* ) const;

        /// block the thread until all asyncronous jobs have completed
        void waitForJobCompletion() const;

    protected:
        /// queue a GPU_Job for submission
        GPU_Job* submit(GPU_Job*);

        /// export specified data to asynchronous tasks in the chain
        //  including the exportComplete() method to clean up
        void exportData( DataBlob* data );
        /// call cleanup associated with the specified data without
        //  propagating to the asyncronous tasks. Typically called 
        //  after error conditions
        void exportCancel( DataBlob* data );

        // will be called immediatley before any chain completion
        // callbacks. Override to clean up any data locks etc.
        // This is where to call the unlock() method.
        // N.B. this should not be called directly. Use exportData()
        // or the exportCancel() methods to ensure mutexs etc are in 
        // place
        virtual void exportComplete( DataBlob* ) = 0;

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
        int unlock( DataBlob* );

    protected:
        static GPU_Manager* gpuManager();

    private:
        ProcessingChain1<DataBlob*>* _chain;

    private:
        void _exportComplete( DataBlob* );
        QHash<const DataBlob*, int> _dataLocker; // keep a track of DataBlobs in use
        mutable QMutex _lockerMutex;
        QList<CallBackT> _linkedFunctors;
        QList<UnlockCallBackT> _unlockTriggers;
        QList<boost::function0<void> > _callbacks; // end of chain callbacks
        QList<DataBlob*> _recentUnlocked;
};

} // namespace lofar
} // namespace pelican
#endif // ASYNCRONOUSMODULE_H 
