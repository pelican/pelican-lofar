#ifndef EMPTY_PIPELINE_H
#define EMPTY_PIPELINE_H


#include "pelican/core/AbstractPipeline.h"
#include "pelican/data/DataBlob.h"
#include "pelican/utility/PelicanTimeRecorder.h"
#include "pelican/output/PelicanTCPBlobServer.h"
#include "TimeSeriesDataSet.h"
#include "SigprocStokesWriter.h"

/**
 * @file EmptyPipeline.h
 */

namespace pelican {
namespace lofar {

/**
 * @class EmptyPipeline
 *
 * @brief
 *
 * @details
 *
 */
class EmptyPipeline : public AbstractPipeline
{
    public:
        EmptyPipeline();
        ~EmptyPipeline();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
        PelicanTimeRecorder _recorder;
        unsigned _iteration;
};

} // namespace lofar
} // namespace pelican

#endif // EMPTY_PIPELINE_H
