#ifndef UDP_BF_PIPELINE_H
#define UDP_BF_PIPELINE_H


#include "pelican/core/AbstractPipeline.h"
#include "pelican/data/DataBlob.h"
#include "pelican/output/PelicanTCPBlobServer.h"

#include "PPFChanneliser.h"
#include "SubbandSpectra.h"
#include "SubbandTimeSeries.h"

/**
 * @file UdpBFPipeline.h
 */

namespace pelican {
namespace lofar {

/**
 * @class UdpBFPipeline
 *
 * @brief
 *
 * @details
 *
 */
class UdpBFPipeline : public AbstractPipeline
{
    public:
        UdpBFPipeline();
        ~UdpBFPipeline();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
        /// Module pointers
        PPFChanneliser* ppfChanneliser;

        /// Local data blob
        SubbandSpectraC32* spectra;
        SubbandTimeSeriesC32* timeSeries;

        unsigned _iteration;
};

} // namespace lofar
} // namespace pelican

#endif // UDP_BF_PIPELINE_H
