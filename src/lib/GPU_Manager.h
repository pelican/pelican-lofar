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
 *    A GPU Reosurce Manager and Job Queue
 * @details
 * 
 */

class GPU_Manager
{

    public:
        GPU_Manager();
        ~GPU_Manager();

        void submit( GPU_Job* job ); 
        void addResource(GPU_Resource* r);
        /// return the number of idle GPU resources
        ///  available
        int freeResources() const;
        /// return the number of jobs that are in the queue
        int jobsQueued() const;

    protected:
        void _matchResources();

    private:
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
