#include "H5CVPipeline.h"
#include "WeightedSpectrumDataSet.h"
#include <iostream>

using std::cout;
using std::endl;

namespace pelican {
namespace lofar {


/**
 * @details
 */
H5CVPipeline::H5CVPipeline( const QString& streamIdentifier ) 
    : AbstractPipeline(), _streamIdentifier(streamIdentifier)
{
    _iteration = 0;
}


/**
 * @details
 */
H5CVPipeline::~H5CVPipeline()
{
}


/**
 * @details
 * Initialises the pipeline.
 *
 * This method is run once on construction of the pipeline.
 */
void H5CVPipeline::init()
{
    ConfigNode c = config( QString("H5Pipeline") );
    _totalIterations= c.getOption("totalIterations", "value", "10000").toInt();    // Create modules
    std::cout << _totalIterations << std::endl;
    ppfChanneliser = (PPFChanneliser *) createModule("PPFChanneliser");
    //    stokesGenerator = (StokesGenerator *) createModule("StokesGenerator");
    //    rfiClipper = (RFI_Clipper *) createModule("RFI_Clipper");
    //    stokesIntegrator = (StokesIntegrator *) createModule("StokesIntegrator");

    // Create local datablobs
    spectra = (SpectrumDataSetC32*) createBlob("SpectrumDataSetC32");
    //    stokes = (SpectrumDataSetStokes*) createBlob("SpectrumDataSetStokes");
    //    intStokes = (SpectrumDataSetStokes*) createBlob("SpectrumDataSetStokes");
    //    weightedIntStokes = (WeightedSpectrumDataSet*) createBlob("WeightedSpectrumDataSet");

    // Request remote data
    requestRemoteData(_streamIdentifier);

}

/**
 * @details
 * Runs the pipeline.
 *
 * This method is run repeatedly by the pipeline application every time
 * data matching the requested remote data is available until either
 * the pipeline application is killed or the method 'stop()' is called.
 */
void H5CVPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointer to the remote time series data blob.
    // This is a block of data containing a number of time series of length
    // N for each sub-band and polarisation.
    timeSeries = (TimeSeriesDataSetC32*) remoteData[_streamIdentifier];
    dataOutput( timeSeries, _streamIdentifier);

    // Run the polyphase channeliser.
    // Generates spectra from a blocks of time series indexed by sub-band
    // and polarisation.
    ppfChanneliser->run(timeSeries, spectra);
    dataOutput( spectra, "ComplexVoltageSpectra");
    // Convert spectra in X, Y polarisation into spectra with stokes parameters.
    //    stokesGenerator->run(spectra, stokes);
    // Clips RFI and modifies blob in place
    //    weightedIntStokes->reset(stokes);

    //    rfiClipper->run(weightedIntStokes);
    //    dataOutput(&(weightedIntStokes->stats()), "RFI_Stats");

    //    stokesIntegrator->run(stokes, intStokes);

    // Calls output stream managed->send(data, stream) the output stream
    // manager is configured in the xml.
    //     dataOutput(intStokes, "SpectrumDataSetStokes");

//    stop();

    if (_iteration % 100 == 0)
      cout << "Finished the CV beamforming pipeline, iteration " << _iteration << " out of " << _totalIterations << endl;

    _iteration++;

    if (_iteration == _totalIterations) stop();
}

} // namespace lofar
} // namespace pelican
