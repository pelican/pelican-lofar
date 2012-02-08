#include "DedispersionPipeline.h"
#include "WeightedSpectrumDataSet.h"
#include "DedispersedTimeSeries.h"
#include <boost/bind.hpp>


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
    delete _dedispersedDataBuffer;
    foreach(SpectrumDataSetStokes* d, _stokesData ) {
        delete d;
    }
    foreach(DedispersionSpectra* d, _dedispersedData ) {
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
     _dedispersionModule = (DedispersionModule*) createModule("DedispersionModule");
     _dedispersionAnalyser = (DedispersionAnalyser*) createModule("DedispersionAnalyser");
     _dedispersionModule->connect( boost::bind( &DedispersionAnalyser::run, _dedispersionAnalyser, _1 ) );
     _dedispersionModule->onChainCompletion( boost::bind( &DedispersionPipeline::updateBufferLock, this ) );

    // Create local datablobs
    spectra = (SpectrumDataSetC32*) createBlob("SpectrumDataSetC32");
    _stokesData = createBlobs<SpectrumDataSetStokes>("SpectrumDataSetStokes", history);
    _stokesBuffer = new LockingCircularBuffer<SpectrumDataSetStokes*>(&_stokesData);
    _dedispersedData = createBlobs<DedispersionSpectra >("DedispersionSpectra", history);
    _dedispersedDataBuffer = new LockingCircularBuffer<DedispersionSpectra* >(&_dedispersedData);

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
    SpectrumDataSetStokes* stokes=_stokesBuffer->next();
    _stokesGenerator->run(spectra, stokes);

    // Clips RFI and modifies blob in place
    weightedIntStokes->reset(stokes);

    _rfiClipper->run(weightedIntStokes);
    dataOutput(&(weightedIntStokes->stats()), "RFI_Stats");

    // start the asyncronous chain of events
    _dedispersionModule->dedisperse(weightedIntStokes, _dedispersedDataBuffer );

}

void DedispersionPipeline::updateBufferLock( ) {
     _stokesBuffer->shiftLock();
}

} // namespace lofar
} // namespace pelican
