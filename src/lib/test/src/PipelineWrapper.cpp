#include "PipelineWrapper.h"
#include "pelican/core/PipelineApplication.h"
#include "pelican/core/AbstractPipeline.h"


namespace pelican {

namespace lofar {


/**
 *@details PipelineWrapper 
 */
PipelineWrapper::PipelineWrapper( AbstractPipeline* pipeline, PipelineApplication* app)
    :  _pipeline(pipeline), _app(app)
{
    timerInit(&_runTime);
}

/**
 *@details
 */
PipelineWrapper::~PipelineWrapper()
{
}

void PipelineWrapper::init() {
    copyConfig(_pipeline);
    _pipeline->init();
}

void PipelineWrapper::run( QHash<QString, DataBlob*>& data ) {
     timerStart(&_runTime);
     _pipeline->run(data);
     timerUpdate(&_runTime);
     _app->stop();
     timerReport(&_runTime, "Pipeline Time: run()");
}

} // namespace lofar
} // namespace pelican
