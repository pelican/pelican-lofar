#include "AsyncronousJob.h"


namespace pelican {

namespace lofar {


/**
 *@details AsyncronousJob 
 */
AsyncronousJob::AsyncronousJob( AsyncronousTask::DataBlobFunctorMonitor* functor )
    : _functor(functor)
{
}

/**
 *@details
 */
AsyncronousJob::~AsyncronousJob()
{
}

} // namespace lofar
} // namespace pelican
