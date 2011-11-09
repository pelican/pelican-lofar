#ifndef ASYNCRONOUSJOB_H
#define ASYNCRONOUSJOB_H

#include "AsyncronousTask.h"

/**
 * @file AsyncronousJob.h
 */

namespace pelican {
class DataBlob;

namespace lofar {

/**
 * @class AsyncronousJob
 *  
 * @brief
 *    Base class for all asyncronous jobs
 * @details
 *
 */

class AsyncronousJob
{
    public:
        AsyncronousJob( AsyncronousTask::DataBlobFunctorMonitor* functor );
        virtual ~AsyncronousJob();

        virtual void submit() = 0;

    protected:
        AsyncronousTask::DataBlobFunctorMonitor* _functor;
};

} // namespace lofar
} // namespace pelican
#endif // ASYNCRONOUSJOB_H 
