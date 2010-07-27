#include "EmptyPipeline.h"
#include <iostream>

namespace pelican {
namespace lofar {


/**
 * @details EmptyPipeline
 */
EmptyPipeline::EmptyPipeline()
    : AbstractPipeline()
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
    // Request remote data
    requestRemoteData("SubbandTimeSeriesC32");
}

/**
 * @details
 * Runs the pipeline.
 */
void EmptyPipeline::run(QHash<QString, DataBlob*>& remoteData)
{

    if (_iteration % 200 == 0) std::cout << "Finished pipeline, iteration " << _iteration << std::endl;
    _iteration++;
}

} // namespace lofar
} // namespace pelican
