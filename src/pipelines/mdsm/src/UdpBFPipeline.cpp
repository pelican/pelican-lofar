#include "UdpBFPipeline.h"
#include <iostream>

namespace pelican {
namespace lofar {


/**
 * @details UdpBFPipeline
 */
UdpBFPipeline::UdpBFPipeline()
    : AbstractPipeline()
{
    _iteration = 0;
}

/**
 * @details
 */
UdpBFPipeline::~UdpBFPipeline()
{
}

/**
 * @details
 * Initialises the pipeline.
 */
void UdpBFPipeline::init()
{
    // Create modules
    ppfChanneliser = (PPFChanneliser *) createModule("PPFChanneliser");

    // Create local datablobs
    spectra = (SubbandSpectraC32*) createBlob("SubbandSpectraC32");

    // Request remote data
    requestRemoteData("SubbandTimeSeriesC32");
}

/**
 * @details
 * Runs the pipeline.
 */
void UdpBFPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointer to the remote time series data blob.
    // Note: This contains the time series data in blocks of nChannels for
    // a number of subbands, polarisations and blocks.
    timeSeries = (SubbandTimeSeriesC32 *) remoteData["SubbandTimeSeriesC32"];

    // Run the polyphase channeliser.
    // Note: This channelises all of the subbands, and polarisations in the time series for
    // a number of blocks of spectra.
    ppfChanneliser->run(timeSeries, spectra);

    // Output channelised data blob (which has dimensions: number of spectra x subbands x polarisations)
    dataOutput(spectra, "SubbandSpectraC32");

    if (_iteration % 200 == 0) std::cout << "Finished the UDP beamforming pipeline, iteration " << _iteration << std::endl;
    _iteration++;
}

} // namespace lofar
} // namespace pelican
