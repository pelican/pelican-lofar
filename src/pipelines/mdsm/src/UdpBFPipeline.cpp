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
    stokesGenerator = (StokesGenerator *) createModule("StokesGenerator");
    stokesIntegrator = (StokesIntegrator *) createModule("StokesIntegrator");
    // Create local datablobs
    spectra = (SpectrumDataSetC32*) createBlob("SubbandSpectraC32");
    stokes = (SpectrumDataSetStokes*) createBlob("SubbandSpectraStokes");
    intStokes = (SpectrumDataSetStokes*) createBlob("SubbandSpectraStokes");
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
    timeSeries = (TimeSeriesDataSetC32*) remoteData["SubbandTimeSeriesC32"];

//    unsigned block = 0, subband = 0, pol = 0;
//    std::cout << "--------- iteration = " << _iteration << " -------- "<<  std::endl;
//    std::complex<float>* t = timeSeries->ptr(block, subband, pol)->ptr();
//    std::cout << "t[0] = " << t[0] << " (b=" << block << ", s=" << subband << ", p=" << pol << ")" << std::endl;
//    std::cout << "t[1] = " << t[1] << " (b=" << block << ", s=" << subband << ", p=" << pol << ")" << std::endl;
//    std::cout << "t[2] = " << t[2] << " (b=" << block << ", s=" << subband << ", p=" << pol << ")" << std::endl;
//    std::cout << std::endl;
//
//    block = 0, subband = 2, pol = 0;
//    t = timeSeries->ptr(block, subband, pol)->ptr();
//    std::cout << "t[0] = " << t[0] << " (b=" << block << ", s=" << subband << ", p=" << pol << ")" << std::endl;
//    std::cout << "t[1] = " << t[1] << " (b=" << block << ", s=" << subband << ", p=" << pol << ")" << std::endl;
//    std::cout << "t[2] = " << t[2] << " (b=" << block << ", s=" << subband << ", p=" << pol << ")" << std::endl;
//    std::cout << "t[511] = " << t[511] << " (b=" << block << ", s=" << subband << ", p=" << pol << ")" << std::endl;
//    std::cout << std::endl;
//
//    block = 0, subband = 2, pol = 1;
//    t = timeSeries->ptr(block, subband, pol)->ptr();
//    std::cout << "t[0] = " << t[0] << " (b=" << block << ", s=" << subband << ", p=" << pol << ")" << std::endl;
//    std::cout << "t[1] = " << t[1] << " (b=" << block << ", s=" << subband << ", p=" << pol << ")" << std::endl;
//    std::cout << "t[2] = " << t[2] << " (b=" << block << ", s=" << subband << ", p=" << pol << ")" << std::endl;
//    std::cout << "--------------------------" << std::endl << std::endl;;


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

    if (_iteration % 1 == 0) std::cout << "Finished the UDP beamforming pipeline, iteration " << _iteration << std::endl;
    _iteration++;
    if (_iteration > 43000) stop();
}

} // namespace lofar
} // namespace pelican
