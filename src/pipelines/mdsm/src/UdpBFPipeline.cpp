#include "UdpBFPipeline.h"
#include <iostream>

using std::cout;
using std::endl;

namespace pelican {
namespace lofar {


/**
 * @details
 */
UdpBFPipeline::UdpBFPipeline() : AbstractPipeline()
{
    _iteration = 0;

    // Initialise timer data.
    timerInit(&_ppfTime);
    timerInit(&_stokesTime);
    timerInit(&_integratorTime);
    timerInit(&_outputTime);
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
 *
 * This method is run once on construction of the pipeline.
 */
void UdpBFPipeline::init()
{
    // Create modules
    ppfChanneliser = (PPFChanneliser *) createModule("PPFChanneliser");
    stokesGenerator = (StokesGenerator *) createModule("StokesGenerator");
    stokesIntegrator = (StokesIntegrator *) createModule("StokesIntegrator");

    // Create local datablobs
    spectra = (SpectrumDataSetC32*) createBlob("SpectrumDataSetC32");
    stokes = (SpectrumDataSetStokes*) createBlob("SpectrumDataSetStokes");
    intStokes = (SpectrumDataSetStokes*) createBlob("SpectrumDataSetStokes");

    // Request remote data
    requestRemoteData("TimeSeriesDataSetC32");

}

/**
 * @details
 * Runs the pipeline.
 *
 * This method is run repeatedly by the pipeline application every time
 * data matching the requested remote data is available until either
 * the pipeline application is killed or the method 'stop()' is called.
 */
void UdpBFPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointer to the remote time series data blob.
    // This is a block of data containing a number of time series of length
    // N for each sub-band and polarisation.
    timeSeries = (TimeSeriesDataSetC32*) remoteData["TimeSeriesDataSetC32"];

    // Get the total number of samples per chunk.
    _totalSamplesPerChunk =
    		timeSeries->nTimesPerBlock() * timeSeries->nTimeBlocks();

    // Run the polyphase channeliser.
    // Generates spectra from a blocks of time series indexed by sub-band
    // and polarisation.
    timerStart(&_ppfTime);
    ppfChanneliser->run(timeSeries, spectra);
    timerUpdate(&_ppfTime);

    // Convert spectra in X, Y polarisation into spectra with stokes parameters.
    timerStart(&_stokesTime);
    stokesGenerator->run(spectra, stokes);
    timerUpdate(&_stokesTime);

    // timerStart(&_integratorTime);
    // stokesIntegrator->run(stokes, intStokes);
    // timerUpdate(&_integratorTime);

    // Calls output stream managed->send(data, stream) the output stream
    // manager is configured in the xml.
    //dataOutput(spectra, "SpectrumDataSetC32");
    timerStart(&_outputTime);
    dataOutput(stokes, "SpectrumDataSetStokes");
    timerUpdate(&_outputTime);

//    stop();

    if (_iteration % 1 == 0)
        cout << "Finished the UDP beamforming pipeline, iteration " << _iteration << endl;

    _iteration++;

//    if (_iteration > 43000) stop();
    if (_iteration * _totalSamplesPerChunk > 20e6) {
    	stop();
    	timerReport(&_ppfTime, "Polyphase Filter");
    	timerReport(&_stokesTime, "Stokes Generator");
    	timerReport(&_integratorTime, "Stokes Integrator");
    	timerReport(&_outputTime, "Output");
    }
}

} // namespace lofar
} // namespace pelican
