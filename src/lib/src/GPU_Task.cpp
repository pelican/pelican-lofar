#include "GPU_Task.h"
#include "GPU_Manager.h"
#include "GPU_Job.h"
#include <boost/bind.hpp>


namespace pelican {

namespace lofar {

/**
 *@details GPU_Task 
 */
GPU_Task::GPU_Task( GPU_Manager* manager, const preprocessingFunctorT& pre, const postprocessingFunctorT& post )
    : AsyncronousTask( boost::bind( &GPU_Task::runJob, this, _1 ) ), _manager(manager)
{
    _pre = pre;
    _postProcessingTask = post;
}

/**
 *@details
 */
GPU_Task::~GPU_Task()
{
}

DataBlob* GPU_Task::runJob( DataBlob* data ) {
     GPU_Job job;
     _pre(&job, data);
     _manager->submit(&job);
     job.wait();
     return _postProcessingTask(&job);
}


} // namespace lofar
} // namespace pelican
