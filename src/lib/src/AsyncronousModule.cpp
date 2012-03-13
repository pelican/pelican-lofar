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
     QList< boost::function0<void> > functors;
     // not very efficient - better to create a ProcessChain taking a single arg
     foreach( const CallBackT& functor,_linkedFunctors ) {
        functors.append( boost::bind( &CallBackT::operator(), &functor, data ) );
     }
     QList<boost::function0<void> > callbacks;
     callbacks << boost::bind( &AsyncronousModule::_exportComplete, this, data) << _callbacks;
     _chain.exec(functors, callbacks );
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

/*
void AsyncronousModule::exportData( DataBlob* data ) {
    // pass resultant data down to any linked tasks
     if( _linkedFunctors.size() == 0 ) {
        // if there are no dependent tasks indicate we have finished
        _finished();
     }
     else {
        // each task is launched in a separate thread
        foreach( const CallBackT& functor, _linkedFunctors ) { 
            ++_dataLocker[ data ];
            QtConcurrent::run( this, &AsyncronousModule::_runTask, functor, data );
        }
     }
}
*/

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

int AsyncronousModule::unlock( const DataBlob* data ) {
    Q_ASSERT( _dataLocker[data] > 0 );
    if( --_dataLocker[data] == 0 ) {
        _recentUnlocked.append(data);
    }
    return _dataLocker[data];
}

void AsyncronousModule::_runTask( const CallBackT& functor, DataBlob* data ) {
     functor(data);
     if( --_dataLocker[data] == 0) {
        _dataLocker.remove(data);
        _finished();
     }
}

void AsyncronousModule::_finished()
{
    // call each post chain event sequentialy
    foreach( const boost::function0<void>& fn, _callbacks ) {
        fn();
    }
}

} // namespace lofar
} // namespace pelican
