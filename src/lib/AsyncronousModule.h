#ifndef ASYNCRONOUSMODULE_H
#define ASYNCRONOUSMODULE_H


#include "pelican/core/AbstractModule.h"
#include "AsyncronousTask.h"

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

class AsyncronousModule : public AbstractModule, public AsyncronousTask
{
    public:
        AsyncronousModule( const ConfigNode& config );
        virtual ~AsyncronousModule();

    protected:
        /// queue a GPU_Job for submission
        GPU_Job* submit(GPU_Job*);

    private:
        static GPU_Manager* gpuManager();

};

} // namespace lofar
} // namespace pelican
#endif // ASYNCRONOUSMODULE_H 
