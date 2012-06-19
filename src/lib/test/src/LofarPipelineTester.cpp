#include "LofarPipelineTester.h"
#include "pelican/core/AbstractPipeline.h"
#include "pelican/core/PipelineApplication.h"
#include "LofarDataBlobGenerator.h"


namespace pelican {

namespace lofar {


/**
 *@details LofarPipelineTester 
 */
void LofarPipelineTester::_init( const QString& configXML )
{
    // ensure initialised form the passed configuration
    _config.setXML( configXML );
    _app = new PipelineApplication(_config);
     QString generator = "LofarDataBlobGenerator";
     // add sutable stanza to config
     QString genXML =  "<" + generator + ">"
                       "</" + generator + ">";
    _app->setDataClient( generator );
}

/**
 *@details
 */
LofarPipelineTester::~LofarPipelineTester()
{
     delete _app;
}

void LofarPipelineTester::run( unsigned iterations )
{
     Q_ASSERT( ! _app->isRunning() );
     _pipeline->setIterations( iterations );
     _app->start(); // pipeline wrapper will stop()
}

} // namespace lofar
} // namespace pelican
