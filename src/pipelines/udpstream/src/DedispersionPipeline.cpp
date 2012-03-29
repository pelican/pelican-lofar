#include "DedispersionPipeline.h"
#include "WeightedSpectrumDataSet.h"
#include "DedispersedTimeSeries.h"
#include "DedispersionDataAnalysis.h"
#include <boost/bind.hpp>
#include "SpectrumDataSet.h"


namespace pelican {

namespace lofar {


/**
 *@details DedispersionPipeline 
 */
DedispersionPipeline::DedispersionPipeline( const QString& streamIdentifier )
    : AbstractPipeline(), _streamIdentifier(streamIdentifier)
{
     _stokesBuffer = 0;
     _dedispersedDataBuffer = 0;
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
    ConfigNode c = config( QString("DedispersionPipeline") );
    unsigned int history= c.getOption("history", "value", "10").toUInt();

    // Create modules
    _ppfChanneliser = (PPFChanneliser *) createModule("PPFChanneliser");
    _stokesGenerator = (StokesGenerator *) createModule("StokesGenerator");
    _rfiClipper = (RFI_Clipper *) createModule("RFI_Clipper");
    _stokesIntegrator = (StokesIntegrator *) createModule("StokesIntegrator");
     _dedispersionModule = (DedispersionModule*) createModule("DedispersionModule");
     _dedispersionAnalyser = (DedispersionAnalyser*) createModule("DedispersionAnalyser");
     _dedispersionModule->connect( boost::bind( &DedispersionPipeline::dedispersionAnalysis, this, _1 ) );
     _dedispersionModule->unlockCallback( boost::bind( &DedispersionPipeline::updateBufferLock, this, _1 ) );

    // Create local datablobs
    spectra = (SpectrumDataSetC32*) createBlob("SpectrumDataSetC32");
    _stokesData = createBlobs<SpectrumDataSetStokes>("SpectrumDataSetStokes", history);
    _stokesBuffer = new LockingPtrContainer<SpectrumDataSetStokes>(&_stokesData);
    _dedispersedData = createBlobs<DedispersionSpectra >("DedispersionSpectra", history);
    _dedispersedDataBuffer = new LockingPtrContainer<DedispersionSpectra>(&_dedispersedData);

    _weightedData = createBlobs<WeightedSpectrumDataSet>("WeightedSpectrumDataSet", history );
    _weightedDataBuffer = new LockingPtrContainer<WeightedSpectrumDataSet>(&_weightedData);


    // Request remote data
    requestRemoteData( _streamIdentifier, history );

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

    // get the next suitable datablob
    WeightedSpectrumDataSet* weightedIntStokes = _weightedDataBuffer->next();
    weightedIntStokes->reset(stokes);

    // Clips RFI and modifies blob in place
    _rfiClipper->run(weightedIntStokes);
    dataOutput(&(weightedIntStokes->stats()), "RFI_Stats");

    // start the asyncronous chain of events
    _dedispersionModule->dedisperse(weightedIntStokes, _dedispersedDataBuffer );

}

void DedispersionPipeline::dedispersionAnalysis( DataBlob* blob ) {
    DedispersionDataAnalysis result;
    DedispersionSpectra* data = static_cast<DedispersionSpectra*>(blob);
    if ( _dedispersionAnalyser->analyse(data, &result) )
    {
        dataOutput( &result );
    }
}

void DedispersionPipeline::updateBufferLock( const QList<DataBlob*>& freeData ) {
     // find WeightedDataBlobs that can be unlocked
     foreach( DataBlob* blob, freeData ) {
        Q_ASSERT( blob->type() == "WeightedSpectrumDataSet" );
        const WeightedSpectrumDataSet* d = static_cast<const WeightedSpectrumDataSet*>(blob);
        _stokesBuffer->unlock( static_cast<SpectrumDataSetStokes*>(d->dataSet()) );
        _weightedDataBuffer->unlock( d );
     }
}

} // namespace lofar
} // namespace pelican
