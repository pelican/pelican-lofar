#include "GPU_Manager.h"


namespace pelican {

namespace lofar {


/**
 *@details GPU_Manager 
 */
GPU_Manager::GPU_Manager(QObject* parent)
    : QThread(parent)
{
}

/**
 *@details
 */
GPU_Manager::~GPU_Manager()
{
}

void GPU_Manager::exec() {
     // Thread staring up here
     // Instantiate the cards

     // Start processing the queue
     while( 1 ) {
        if( _queue.size() > 0 ) {
            runJob(_queue.takeFirst() );
        }
     }
}

void GPU_Manager::runJob(const GPU_Job& job)
{
     cudamemcpy( job.memStart(), job.memSize() );
     foreach( const GPU_Kernel&, job.kernels() ) {
        kernel.run();
     }
}

} // namespace lofar
} // namespace pelican
