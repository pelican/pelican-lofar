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
     _dedispersionModule = 0;
     _dedispersionAnalyser = 0;
     _ppfChanneliser = 0;
     _rfiClipper = 0;
     _stokesIntegrator = 0;
     _stokesGenerator = 0;

    // Initialise timer data.
    timerInit(&_ppfTime);
    timerInit(&_rfiClipper);
    timerInit(&_stokesTime);
    timerInit(&_integratorTime);
    timerInit(&_dedispersionTime);
    timerInit(&_outputTime);
    timerInit(&_totalTime);
#ifdef TIMING_ENABLED
    _iteration = 0;
#endif // TIMING_ENABLED
}

/**
 *@details
 */
DedispersionPipeline::~DedispersionPipeline()
{
    delete _dedispersionModule;
    delete _dedispersionAnalyser;
    delete _stokesBuffer;
    delete _ppfChanneliser;
    delete _rfiClipper;
    delete _stokesIntegrator;
    delete _stokesGenerator;

    foreach(SpectrumDataSetStokes* d, _stokesData ) {
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
    _weightedIntStokes = (WeightedSpectrumDataSet*) createBlob("WeightedSpectrumDataSet");

    // Request remote data
    requestRemoteData( _streamIdentifier, 1 );
}

void DedispersionPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    timerStart(&_totalTime);

    // Get pointer to the remote time series data blob.
    // This is a block of data containing a number of time series of length
    // N for each sub-band and polarisation.
    timeSeries = (TimeSeriesDataSetC32*) remoteData[_streamIdentifier];
    dataOutput( timeSeries, _streamIdentifier);

    // Run the polyphase channeliser.
    // Generates spectra from a blocks of time series indexed by sub-band
    // and polarisation.
    timerStart(&_ppfTime);
    _ppfChanneliser->run(timeSeries, _spectra);
    timerUpdate(&_ppfTime);

    // Convert spectra in X, Y polarisation into spectra with stokes parameters.
    timerStart(&_stokesTime);
    SpectrumDataSetStokes* stokes=_stokesBuffer->next();
    _stokesGenerator->run(_spectra, stokes);
    timerUpdate(&_stokesTime);

    // set up a suitable datablob fro the rfi clipper
    _weightedIntStokes->reset(stokes);

    // Clips RFI and modifies blob in place
    timerStart(&_rfiClipper);
    _rfiClipper->run(_weightedIntStokes);
    dataOutput(&(_weightedIntStokes->stats()), "RFI_Stats");
    timerUpdate(&_rfiClipper);

    timerStart(&_integratorTime);
    _stokesIntegrator->run(stokes, _intStokes);
    dataOutput(_intStokes, "SpectrumDataSetStokes");
    timerUpdate(&_integratorTime);

    // start the asyncronous chain of events
    timerStart(&_dedispersionTime);
    _dedispersionModule->dedisperse( _weightedIntStokes );
    timerUpdate(&_dedispersionTime);

#ifdef TIMING_ENABLED
    timerUpdate(&_totalTime);
    if( ++_iteration%2000 == 0 ) {
        timerReport(&adapterTime, "Adapter Time");
        timerReport(&_ppfTime, "Polyphase Filter");
        timerReport(&_stokesTime, "Stokes Generator");
        timerReport(&_rfiClipper, "RFI_Clipper");
        //    	timerReport(&_integratorTime, "Stokes Integrator");
        timerReport(&_outputTime, "Output");
        timerReport(&_totalTime, "Pipeline Time (excluding adapter)");
        std::cout << endl;
        std::cout << "Total (average) allowed time per iteration = "
            << _totalSamplesPerChunk * 5.12e-6 << " sec" << "\n";
        std::cout << "Total (average) actual time per iteration = "
            << adapterTime.timeAverage + _totalTime.timeAverage << " sec" << "\n";
        std::cout << std::endl;
    }
#endif
}

void DedispersionPipeline::dedispersionAnalysis( DataBlob* blob ) {
//qDebug() << "analysis()";
    DedispersionDataAnalysis result;
    DedispersionSpectra* data = static_cast<DedispersionSpectra*>(blob);
    if ( _dedispersionAnalyser->analyse(data, &result) )
    {
        dataOutput( &result, "DedispersionDataAnalysis" );
        foreach( const SpectrumDataSetStokes* d, result.data()->inputDataBlobs()) {
            dataOutput( d, "SignalFoundSpectrum" );
        }
    }
}

void DedispersionPipeline::updateBufferLock( const QList<DataBlob*>& freeData ) {
     // find WeightedDataBlobs that can be unlocked
     foreach( DataBlob* blob, freeData ) {
        Q_ASSERT( blob->type() == "SpectrumDataSetStokes" );
        _stokesBuffer->unlock( static_cast<SpectrumDataSetStokes*>(blob) );
     }
}

} // namespace lofar
} // namespace pelican
