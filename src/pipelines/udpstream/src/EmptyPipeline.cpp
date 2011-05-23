#include "EmptyPipeline.h"
#include <iostream>

using std::cout;
using std::endl;

namespace pelican {
namespace lofar {


/**
 * @details EmptyPipeline
 */
EmptyPipeline::EmptyPipeline() : AbstractPipeline()
{
    _iteration = 0;
}

/**
 * @details
 */
EmptyPipeline::~EmptyPipeline()
{
}


/**
 * @details
 * Initialises the pipeline.
 */
void EmptyPipeline::init()
{
    requestRemoteData("TimeSeriesDataSetC32");
    _recorder.setReportInterval(10000);
    _recorder.start();
}

/**
 * @details
 * Runs the pipeline.
 */
void EmptyPipeline::run(QHash<QString, DataBlob*>& /*remoteData*/)
{
    _recorder.tick("run");
}

} // namespace lofar
} // namespace pelican
