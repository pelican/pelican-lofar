#ifndef PIPELINEWRAPPER_H
#define PIPELINEWRAPPER_H


#include "pelican/core/AbstractPipeline.h"
#include "pelican/core/PipelineApplication.h"
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

template<class AbstractPipelineType>
class PipelineWrapper : public AbstractPipelineType
{
    public:
        PipelineWrapper( AbstractPipelineType* pipeline, PipelineApplication* app ) 
            : AbstractPipelineType(*pipeline), _app_(app) 
            {
                timerInit(&_runTime_);
            }
        virtual ~PipelineWrapper() {};
        virtual void run( QHash<QString, DataBlob*>& data ) {
            timerStart(&_runTime_);
            this->AbstractPipelineType::run(data);
            timerUpdate(&_runTime_);
            _app_->stop();
            timerReport(&_runTime_, "Pipeline Time: run()");
        }

    private:
        PipelineApplication* _app_;
        TimerData _runTime_;
};

} // namespace lofar
} // namespace pelican
#endif // PIPELINEWRAPPER_H 
