#include "DedispersionPipeline.h"
#include "WeightedSpectrumDataSet.h"


namespace pelican {

namespace lofar {


/**
 *@details DedispersionPipeline 
 */
DedispersionPipeline::DedispersionPipeline( const QString& streamIdentifier )
    : AbstractPipeline(), _streamIdentifier(streamIdentifier)
{
}

/**
 *@details
 */
DedispersionPipeline::~DedispersionPipeline()
{
    delete _stokesBuffer;
    foreach(SpectrumDataSetStokes* d, _stokesData ) {
        delete d;
    }
}

void DedispersionPipeline::init()
{
    unsigned int history=10;

    // Create modules
    _ppfChanneliser = (PPFChanneliser *) createModule("PPFChanneliser");
    _stokesGenerator = (StokesGenerator *) createModule("StokesGenerator");
    _rfiClipper = (RFI_Clipper *) createModule("RFI_Clipper");
    _stokesIntegrator = (StokesIntegrator *) createModule("StokesIntegrator");

    // Create local datablobs
    spectra = (SpectrumDataSetC32*) createBlob("SpectrumDataSetC32");
    _stokesData = createBlobs<SpectrumDataSetStokes>("SpectrumDataSetStokes", history);
    _stokesBuffer = new LockingCircularBuffer<SpectrumDataSetStokes*>(&_stokesData);
    intStokes = (SpectrumDataSetStokes*) createBlob("SpectrumDataSetStokes");
    
    weightedIntStokes = (WeightedSpectrumDataSet*) createBlob("WeightedSpectrumDataSet");

    // Request remote data
    requestRemoteData(_streamIdentifier, history );

}

void DedispersionPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointer to the remote time series data blob.
    // This is a block of data containing a number of time series of length
    // N for each sub-band and polarisation.
    timeSeries = (TimeSeriesDataSetC32*) remoteData[_streamIdentifier];
    dataOutput( timeSeries, _streamIdentifier);

    // Run the polyphase channeliser.
    // Generates spectra from a blocks of time series indexed by sub-band
    // and polarisation.
    _ppfChanneliser->run(timeSeries, spectra);

    // Convert spectra in X, Y polarisation into spectra with stokes parameters.
    stokes=_stokesBuffer->next();
    _stokesGenerator->run(spectra, stokes);
    // Clips RFI and modifies blob in place
    weightedIntStokes->reset(stokes);

    _rfiClipper->run(weightedIntStokes);
    dataOutput(&(weightedIntStokes->stats()), "RFI_Stats");

}

} // namespace lofar
} // namespace pelican
