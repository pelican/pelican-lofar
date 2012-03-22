#include "LofarPipelineTester.h"
#include "pelican/core/AbstractPipeline.h"
#include "pelican/core/PipelineApplication.h"
#include "pelican/utility/Config.h"
#include "PipelineWrapper.h"


namespace pelican {

namespace lofar {


/**
 *@details LofarPipelineTester 
 */
LofarPipelineTester::LofarPipelineTester( AbstractPipeline* pipeline, const QString& configXML )
    : _pipeline(0)
{
    // ensure initialised form the passed configuration
    Config config;
    config.setXML( configXML );
    _app = new PipelineApplication(config);

    _pipeline = new PipelineWrapper( pipeline, _app );
    _app->registerPipeline(_pipeline);
    //_app->setDataClient("FileDataClient");

}

/**
 *@details
 */
LofarPipelineTester::~LofarPipelineTester()
{
     delete _pipeline;
     delete _app;
}

void LofarPipelineTester::run()
{
     Q_ASSERT( ! _app->isRunning() );
     _app->start(); // pipeline wrapper will stop()
}

} // namespace lofar
} // namespace pelican
