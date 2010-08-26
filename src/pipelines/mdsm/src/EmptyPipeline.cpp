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
}

/**
 * @details
 * Runs the pipeline.
 */
void EmptyPipeline::run(QHash<QString, DataBlob*>& /*remoteData*/)
{
    if (_iteration % 200 == 0)
        cout << "Finished pipeline, iteration " << _iteration << endl;

    _iteration++;
}

} // namespace lofar
} // namespace pelican
