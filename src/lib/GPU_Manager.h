#ifndef GPU_MANAGER_H
#define GPU_MANAGER_H


#include <QtCore/QList>
#include <QtCore/QSet>
#include <QtCore/QMutex>
#include "GPU_Resource.h"

/**
 * @file GPU_Manager.h
 */

namespace pelican {
namespace lofar {
class GPU_Resource;
class GPU_Job;

/**
 * @class GPU_Manager
 *  
 * @brief
 *    A GPU Resource Manager and Job Queue
 * @details
 *    Allows you to submit jobs to a queue of GPU resources
 *    As the resource becomes available, the next item from the queue
 *    is taken and executed.
 *
 *    Use the @code addResource() method to add GPU cards to be managed. Note that
 *    all the resources should be compatible with the types of jobs to be
 *    submitted - there is no checking of job suitability.
 *
 *    call @code submit() to add a job to be processed. All job status 
 *    information/callbacks etc can be found through the GPU_Job interface.
 *
 */

class GPU_Manager
{

    public:
        GPU_Manager();
        ~GPU_Manager();

        /// submit a job to the queue
        void submit( GPU_Job* job ); 

        /// add a GPU resource (e.g. an NVidia card) to manage
        //  ownership is transferred to the manager
        void addResource(GPU_Resource* r);

        /// return the number of idle GPU resources
        ///  available
        int freeResources() const;

        /// return the number of jobs that are in the queue
        int jobsQueued() const;

    private:
        void _matchResources();
        void _runJob( GPU_Resource*, GPU_Job* );
        void _resourceFree( GPU_Resource* );

    private:
        mutable QMutex _resourceMutex;
        QList<GPU_Job*> _queue;
        QList<GPU_Resource*> _resources;
        QList<GPU_Resource*> _freeResource;
        bool _destructor;

};

} // namespace lofar
} // namespace pelican
#endif // GPU_MANAGER_H 
