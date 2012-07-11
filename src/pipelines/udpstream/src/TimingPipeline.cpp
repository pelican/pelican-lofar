#include "TimingPipeline.h"
#include "AdapterTimeSeriesDataSet.h"
#include "WeightedSpectrumDataSet.h"
#include <iostream>

using std::cout;
using std::endl;

namespace pelican {
namespace lofar {


/**
 * @details
 */
TimingPipeline::TimingPipeline() : AbstractPipeline()
{
    _iteration = 0;

}


/**
 * @details
 */
TimingPipeline::~TimingPipeline()
{
}


/**
 * @details
 * Initialises the pipeline.
 *
 * This method is run once on construction of the pipeline.
 */
void TimingPipeline::init()
{
    // Create modules
    ppfChanneliser = (PPFChanneliser *) createModule("PPFChanneliser");
    stokesGenerator = (StokesGenerator *) createModule("StokesGenerator");
    rfiClipper = (RFI_Clipper *) createModule("RFI_Clipper");
    stokesIntegrator = (StokesIntegrator *) createModule("StokesIntegrator");
    weightedIntStokes = (WeightedSpectrumDataSet*) createBlob("WeightedSpectrumDataSet");

    // Create local datablobs
    spectra = (SpectrumDataSetC32*) createBlob("SpectrumDataSetC32");
    stokes = (SpectrumDataSetStokes*) createBlob("SpectrumDataSetStokes");
    intStokes = (SpectrumDataSetStokes*) createBlob("SpectrumDataSetStokes");
    weightedIntStokes = (WeightedSpectrumDataSet*) createBlob("WeightedSpectrumDataSet");

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
void TimingPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
  timerStart(&_totalTime);

    // Get pointer to the remote time series data blob.
    // This is a block of data containing a number of time series of length
    // N for each sub-band and polarisation.
    timeSeries = (TimeSeriesDataSetC32*) remoteData["LofarTimeStream1"];

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

    // The RFI clipper

    timerStart(&_rfiClipper);
    weightedIntStokes->reset(stokes);
    rfiClipper->run(weightedIntStokes);
    timerUpdate(&_rfiClipper);

    //    timerStart(&_integratorTime);
    //    stokesIntegrator->run(stokes, intStokes);
    //    timerUpdate(&_integratorTime);

    // Calls output stream managed->send(data, stream) the output stream
    // manager is configured in the xml.
    //dataOutput(spectra, "SpectrumDataSetC32");
    timerStart(&_outputTime);
    dataOutput(stokes, "SpectrumDataSetStokes");
    timerUpdate(&_outputTime);

//    stop();

    if (_iteration % 50 == 0)
        cout << "Finished the UDP beamforming pipeline, iteration " << _iteration << endl;
  timerUpdate(&_totalTime);
    ++_iteration;

//    if (_iteration > 43000) stop();
    if (_iteration * _totalSamplesPerChunk >= 16*16384*5) {
        stop();
    //timerReport(&(adapter->timeData()), "Adapter Time");
    timerReport(&_ppfTime, "Polyphase Filter");
    timerReport(&_stokesTime, "Stokes Generator");
    timerReport(&_rfiClipper, "RFI_Clipper");
    //    	timerReport(&_integratorTime, "Stokes Integrator");
    timerReport(&_outputTime, "Output");
    timerReport(&_totalTime, "Pipeline Time (excluding adapter)");
    cout << endl;
    cout << "Total (average) allowed time per iteration = "
         << _totalSamplesPerChunk * 5.12e-6 << " sec" << endl;
    //cout << "Total (average) actual time per iteration = "
    //     << adapterTime.timeAverage + _totalTime.timeAverage << " sec" << endl;
    cout << "nSubbands = " << timeSeries->nSubbands() << endl;
    cout << "nPols = " << timeSeries->nPolarisations() << endl;
    cout << "nBlocks = " << timeSeries->nTimeBlocks() << endl;
    cout << "nChannels = " << timeSeries->nTimesPerBlock() << endl;
    cout << endl;
    }
}

} // namespace lofar
} // namespace pelican
