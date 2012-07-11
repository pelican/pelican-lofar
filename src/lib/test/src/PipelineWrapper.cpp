#include "PipelineWrapper.h"
#include "pelican/core/PipelineApplication.h"
#include "pelican/core/AbstractPipeline.h"
#include <QDebug>


namespace pelican {

namespace lofar {


/**
 *@details PipelineWrapper 
 *  base class for PipelineWrapperSpecialisation
 *  which is the one you should use
 */
PipelineWrapper::PipelineWrapper( PipelineApplication* app)
    :  _app_(app)
{
    //
    _pipelineWrapperIterationCount = 1;
}

void PipelineWrapper::setIterations( unsigned i ) {
    _pipelineWrapperIterationCount = i;
}

} // namespace lofar
} // namespace pelican
