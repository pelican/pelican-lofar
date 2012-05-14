#include "DedispersionPipeline.h"
#include "WeightedSpectrumDataSet.h"
#include "DedispersedTimeSeries.h"
#include "DedispersionDataAnalysis.h"
#include <boost/bind.hpp>
#include "SpectrumDataSet.h"
#include <QDebug>


namespace pelican {

namespace lofar {


/**
 *@details DedispersionPipeline 
 */
DedispersionPipeline::DedispersionPipeline( const QString& streamIdentifier )
    : AbstractPipeline(), _streamIdentifier(streamIdentifier)
{
     _spectra = 0;
     _stokesBuffer = 0;
     _dedispersedDataBuffer = 0;
     _dedispersionModule = 0;
     _dedispersionAnalyser = 0;
     _ppfChanneliser = 0;
     _rfiClipper = 0;
     _stokesIntegrator = 0;
     _stokesGenerator = 0;

}

/**
 *@details
 */
DedispersionPipeline::~DedispersionPipeline()
{
    delete _dedispersionModule;
    delete _dedispersionAnalyser;
    delete _stokesBuffer;
    delete _dedispersedDataBuffer;
    delete _ppfChanneliser;
    delete _rfiClipper;
    delete _stokesIntegrator;
    delete _stokesGenerator;

    foreach(SpectrumDataSetStokes* d, _stokesData ) {
        delete d;
    }
    foreach(DedispersionSpectra* d, _dedispersedData ) {
        delete d;
    }
    delete _spectra;
}

void DedispersionPipeline::init()
{
    ConfigNode c = config( QString("DedispersionPipeline") );
    // history indicates the number of datablobs to keep (iterations of run())
    // it should be Dedidpersion Buffer size (in Blobs)*number of Dedispersion Buffers
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
    _spectra = (SpectrumDataSetC32*) createBlob("SpectrumDataSetC32");
    _stokesData = createBlobs<SpectrumDataSetStokes>("SpectrumDataSetStokes", history);
    _intStokes = (SpectrumDataSetStokes*) createBlob("SpectrumDataSetStokes");
    _stokesBuffer = new LockingPtrContainer<SpectrumDataSetStokes>(&_stokesData);
    _dedispersedData = createBlobs<DedispersionSpectra >("DedispersionSpectra", history);
    _dedispersedDataBuffer = new LockingPtrContainer<DedispersionSpectra>(&_dedispersedData);

    _weightedIntStokes = (WeightedSpectrumDataSet*) createBlob("WeightedSpectrumDataSet");

    // Request remote data
    requestRemoteData( _streamIdentifier, 1 );
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
    _ppfChanneliser->run(timeSeries, _spectra);

    // Convert spectra in X, Y polarisation into spectra with stokes parameters.
    SpectrumDataSetStokes* stokes=_stokesBuffer->next();
    _stokesGenerator->run(_spectra, stokes);

    // set up a suitable datablob fro the rfi clipper
    _weightedIntStokes->reset(stokes);

    // Clips RFI and modifies blob in place
    _rfiClipper->run(_weightedIntStokes);
    dataOutput(&(_weightedIntStokes->stats()), "RFI_Stats");

    _stokesIntegrator->run(stokes, _intStokes);
    dataOutput(_intStokes, "SpectrumDataSetStokes");

    // start the asyncronous chain of events
    _dedispersionModule->dedisperse( _weightedIntStokes, _dedispersedDataBuffer );

}

void DedispersionPipeline::dedispersionAnalysis( DataBlob* blob ) {
//qDebug() << "analysis()";
    DedispersionDataAnalysis result;
    DedispersionSpectra* data = static_cast<DedispersionSpectra*>(blob);
    if ( _dedispersionAnalyser->analyse(data, &result) )
    {
        dataOutput( &result );
        foreach( const SpectrumDataSetStokes* d, result.data()->inputDataBlobs()) {
            dataOutput( d, "SignalFoundSpectrum" );
        }
    }
}

void DedispersionPipeline::updateBufferLock( const QList<DataBlob*>& freeData ) {
     // find WeightedDataBlobs that can be unlocked
//qDebug() << "unlocking()";
     foreach( DataBlob* blob, freeData ) {
        Q_ASSERT( blob->type() == "SpectrumDataSetStokes" );
        _stokesBuffer->unlock( static_cast<SpectrumDataSetStokes*>(blob) );
     }
}

} // namespace lofar
} // namespace pelican
