#ifndef PIPELINEWRAPPER_H
#define PIPELINEWRAPPER_H


#include "pelican/core/AbstractPipeline.h"
#include "timer.h"

/**
 * @file PipelineWrapper.h
 */

namespace pelican {
class AbstractPipeline;
class PipelineApplication;

namespace lofar {

/**
 * @class PipelineWrapper
 *  
 * @brief
 *     Wraps a pipeline to provide timing information and
 *     a run once functionality
 * @details
 *     To invoke the pipeline use the PelicanApplication::start() method.
 *     After running the wrapped pipeline, the stop() method will be called
 * 
 */

class PipelineWrapper : public AbstractPipeline
{
    public:
        PipelineWrapper( AbstractPipeline* pipeline, PipelineApplication* app );
        virtual ~PipelineWrapper();
        virtual void init();
        virtual void run( QHash<QString, DataBlob*>& data );

    private:
        AbstractPipeline* _pipeline;
        PipelineApplication* _app;
        TimerData _runTime;
};

} // namespace lofar
} // namespace pelican
#endif // PIPELINEWRAPPER_H 
