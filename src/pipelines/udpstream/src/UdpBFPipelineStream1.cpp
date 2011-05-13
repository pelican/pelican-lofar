#include "UdpBFPipelineStream1.h"
#include <iostream>

using std::cout;
using std::endl;

namespace pelican {
namespace lofar {


/**
 * @details
 */
UdpBFPipelineStream1::UdpBFPipelineStream1() : AbstractPipeline()
{
    _iteration = 0;
}


/**
 * @details
 */
UdpBFPipelineStream1::~UdpBFPipelineStream1()
{
}


/**
 * @details
 * Initialises the pipeline.
 *
 * This method is run once on construction of the pipeline.
 */
void UdpBFPipelineStream1::init()
{
    // Create modules
    ppfChanneliser = (PPFChanneliser *) createModule("PPFChanneliser");
    stokesGenerator = (StokesGenerator *) createModule("StokesGenerator");
    rfiClipper = (RFI_Clipper *) createModule("RFI_Clipper");
    stokesIntegrator = (StokesIntegrator *) createModule("StokesIntegrator");

    // Create local datablobs
    spectra = (SpectrumDataSetC32*) createBlob("SpectrumDataSetC32");
    stokes = (SpectrumDataSetStokes*) createBlob("SpectrumDataSetStokes");
    intStokes = (SpectrumDataSetStokes*) createBlob("SpectrumDataSetStokes");

    // Request remote data
    requestRemoteData("LofarTimeStream1");

}

/**
 * @details
 * Runs the pipeline.
 *
 * This method is run repeatedly by the pipeline application every time
 * data matching the requested remote data is available until either
 * the pipeline application is killed or the method 'stop()' is called.
 */
void UdpBFPipelineStream1::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointer to the remote time series data blob.
    // This is a block of data containing a number of time series of length
    // N for each sub-band and polarisation.
    timeSeries = (TimeSeriesDataSetC32*) remoteData["LofarTimeStream1"];

    //timeSeries->write("timeStream1-s1-p0.dat", 1, 0, -1);
    //timeSeries->write("timeStream1-i" + QString::number(_iteration) + "-s1-p0.dat", 1, 0, -1);

    // Run the polyphase channeliser.
    // Generates spectra from a blocks of time series indexed by sub-band
    // and polarisation.
    ppfChanneliser->run(timeSeries, spectra);

//    spectra->write("stream1-s0-p0-b0.dat", 0, 0, 0);

    // Convert spectra in X, Y polarisation into spectra with stokes parameters.
    stokesGenerator->run(spectra, stokes);
    // Clips RFI and modifies blob in place
    rfiClipper->run(stokes);

    stokesIntegrator->run(stokes, intStokes);

    // Calls output stream managed->send(data, stream) the output stream
    // manager is configured in the xml.
     dataOutput(intStokes, "SpectrumDataSetStokes");

     //if (_iteration == 5) stop();
    //stop();

    if (_iteration % 100 == 0)
        cout << "Finished the UDP beamforming pipeline, iteration " << _iteration << endl;

    _iteration++;

    if (_iteration > 2500000) stop();
}

} // namespace lofar
} // namespace pelican
