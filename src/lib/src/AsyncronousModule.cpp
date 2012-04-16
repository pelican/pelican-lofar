#include "AsyncronousModule.h"
#include <QtConcurrentRun>
#include "GPU_Manager.h"
#include "GPU_NVidia.h"
#include <boost/bind.hpp>
#include <iostream>


namespace pelican {

namespace lofar {


/**
 *@details AsyncronousModule 
 */
AsyncronousModule::AsyncronousModule( const ConfigNode& config )
    : AbstractModule( config )
{
   _chain = new ProcessingChain1<DataBlob*>;
   // initialise the GPU manager if required
   // for now we share the mamanger between all instances
   // and hog all the cards. We could refine this by removing
   // the static and passing
   // down an appropriately configured gpuManager in the 
   // constructor.
   if( gpuManager()->resources() == 0 ) {
       GPU_NVidia::initialiseResources( gpuManager() );
   }
}

GPU_Manager* AsyncronousModule::gpuManager() {
    static GPU_Manager gpuManager;
    return &gpuManager;
}

/**
 *@details
 */
AsyncronousModule::~AsyncronousModule()
{
    // delete _chain first as this will ensure that all 
    // outstanding jobs
    // are finished before removing the rest of the object
    delete _chain;
}

void AsyncronousModule::connect( const boost::function1<void, DataBlob*>& functor ) {
    _linkedFunctors.append(functor);
}

void AsyncronousModule::unlockCallback( const UnlockCallBackT& callback ) {
    _unlockTriggers.append(callback);
}

GPU_Job* AsyncronousModule::submit(GPU_Job* job) {
    return gpuManager()->submit(job);
}

void AsyncronousModule::exportData( DataBlob* data ) {
     QList<boost::function0<void> > callbacks;
     callbacks << boost::bind( &AsyncronousModule::_exportComplete, this, data) << _callbacks;
     _chain->exec(_linkedFunctors, callbacks, data );
}

void AsyncronousModule::_exportComplete( DataBlob* blob ) {
     // allow derived class space to unlock
     QMutexLocker lock( &_lockerMutex );
     exportComplete( blob );
     // call unlocked triggers
     foreach( const UnlockCallBackT& functor, _unlockTriggers ) { 
        functor( _recentUnlocked );
     }
     _recentUnlocked.clear();
}

void AsyncronousModule::lock( const DataBlob* data ) {
    QMutexLocker lock(&_lockerMutex);
    ++_dataLocker[data];
}

int AsyncronousModule::lockNumber( const DataBlob* data ) const
{
    QMutexLocker lock(&_lockerMutex);
    if( _dataLocker.contains( data ) )
        return _dataLocker.value(data);
    return 0;
}

int AsyncronousModule::unlock( DataBlob* data ) {
    Q_ASSERT( _dataLocker[data] > 0 );
    if( --_dataLocker[data] == 0 ) {
        _recentUnlocked.append(data);
    }
    return _dataLocker[data];
}

} // namespace lofar
} // namespace pelican
