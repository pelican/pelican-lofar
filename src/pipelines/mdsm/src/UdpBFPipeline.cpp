#include "UdpBFPipeline.h"
#include <iostream>

using std::cout;
using std::endl;

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
    stokesGenerator = (StokesGenerator *) createModule("StokesGenerator");
    stokesIntegrator = (StokesIntegrator *) createModule("StokesIntegrator");

    // Create local datablobs
    spectra = (SpectrumDataSetC32*) createBlob("SubbandSpectraC32");
    stokes = (SpectrumDataSetStokes*) createBlob("SubbandSpectraStokes");
    intStokes = (SpectrumDataSetStokes*) createBlob("SubbandSpectraStokes");


    // Request remote data
    requestRemoteData("TimeSeriesDataSetC32");
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
    timeSeries = (TimeSeriesDataSetC32*) remoteData["SubbandSpectraC32"];

    // Run the polyphase channeliser.
    // Note: This channelises all of the subbands, and polarisations in the time series for
    // a number of blocks of spectra.
    ppfChanneliser->run(timeSeries, spectra);

    stokesGenerator->run(spectra, stokes);

    stokesIntegrator->run(stokes, intStokes);

    // Output channelised data blob (which has dimensions: number of spectra x subbands x polarisations)
    //dataOutput(spectra, "SubbandSpectraC32");
    // calls output stream managed->send(data, stream)
    // the output stream manager is configured in the xml

    dataOutput(intStokes, "SubbandSpectraStokes");

//    stop();

    if (_iteration % 1 == 0)
        cout << "Finished the UDP beamforming pipeline, iteration " << _iteration << endl;
    _iteration++;
    if (_iteration > 43000) stop();
}

} // namespace lofar
} // namespace pelican
