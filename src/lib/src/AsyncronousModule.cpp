#include "AsyncronousModule.h"
#include <QtConcurrentRun>
#include "GPU_Manager.h"
#include "GPU_NVidia.h"


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

GPU_Job* AsyncronousModule::submit(GPU_Job* job) {
    return gpuManager()->submit(job);
}

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
