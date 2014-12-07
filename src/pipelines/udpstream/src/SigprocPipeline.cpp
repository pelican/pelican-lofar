#include "DedispersionModule.h"
#include "DedispersionDataAnalysis.h"
#include "SigprocPipeline.h"
#include "SigprocAdapter.h"
#include "SpectrumDataSet.h"
#include "WeightedSpectrumDataSet.h"
#include <boost/bind.hpp>

namespace pelican {

namespace ampp {


/**
 *@details SigprocPipeline 
 */
SigprocPipeline::SigprocPipeline()
    : AbstractPipeline()
{
}

/**
 *@details
 */
SigprocPipeline::~SigprocPipeline()
{
}

void SigprocPipeline::init() {
    _rfiClipper = (RFI_Clipper *) createModule("RFI_Clipper");
    _dedispersionModule = (DedispersionModule*) createModule("DedispersionModule");
    _dedispersionAnalyser = (DedispersionAnalyser*) createModule("DedispersionAnalyser");
    _dedispersionModule->connect( boost::bind( &SigprocPipeline::dedispersionAnalysis, this, _1 ) );
    _dedispersionModule->unlockCallback( boost::bind( &SigprocPipeline::updateBufferLock, this, _1 ) );
    //_stokesIntegrator = (StokesIntegrator *) createModule("StokesIntegrator");
    //_intStokes = (SpectrumDataSetStokes*) createBlob("SpectrumDataSetStokes");
    _weightedIntStokes = (WeightedSpectrumDataSet*) createBlob("WeightedSpectrumDataSet");

    // Request remote data
    requestRemoteData("SpectrumDataSetStokes");
}

void SigprocPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    SpectrumDataSetStokes* stokes = (SpectrumDataSetStokes*)remoteData["SpectrumDataSetStokes"];
    if( ! stokes ) throw(QString("no STOKES!"));
    _weightedIntStokes->reset(stokes);
    _rfiClipper->run(_weightedIntStokes);
    _dedispersionModule->dedisperse(_weightedIntStokes);
    dedispersionAnalysis(stokes);
    //_stokesIntegrator->run(stokes, _intStokes);
    //dataOutput(_intStokes, "SpectrumDataSetStokes");
}

void SigprocPipeline::dedispersionAnalysis( DataBlob* blob ) {
//qDebug() << "analysis()";
//  std::cout << "PIPELINE: in dd analysis" << std::endl;
    DedispersionDataAnalysis result;
    DedispersionSpectra* data = static_cast<DedispersionSpectra*>(blob);
    if ( _dedispersionAnalyser->analyse(data, &result) )
      {
        std::cout << "Found " << result.eventsFound() << " events" << std::endl;
        std::cout << "Limits: " << _minEventsFound << " " << _maxEventsFound << " events" << std::endl;
        dataOutput( &result, "TriggerInput" );
        if (_minEventsFound >= _maxEventsFound){
            std::cout << "Writing out..." << std::endl;
            if (result.eventsFound() >= _minEventsFound){
              dataOutput( &result, "DedispersionDataAnalysis" );
              foreach( const SpectrumDataSetStokes* d, result.data()->inputDataBlobs()) {
                dataOutput( d, "SignalFoundSpectrum" );
                //                    dataOutput( d->getRawData(), "RawDataFoundSpectrum" );
              }
            }
        }
        else{
          if (result.eventsFound() >= _minEventsFound && result.eventsFound() <= _maxEventsFound){
            std::cout << "Writing out..." << std::endl;
            dataOutput( &result, "DedispersionDataAnalysis" );
            foreach( const SpectrumDataSetStokes* d, result.data()->inputDataBlobs()) {
              dataOutput( d, "SignalFoundSpectrum" );
              //                    dataOutput( d->getRawData(), "RawDataFoundSpectrum" );
            }
          }
        }
      }
}

void SigprocPipeline::updateBufferLock( const QList<DataBlob*>& freeData ) {
#if 0
     // find WeightedDataBlobs that can be unlocked
     foreach( DataBlob* blob, freeData ) {
        Q_ASSERT( blob->type() == "SpectrumDataSetStokes" );
        // unlock the pointers to the raw buffer
        // _rawBuffer->unlock( static_cast<SpectrumDataSetStokes*>(blob)->getRawData() );
        _stokesBuffer->unlock( static_cast<SpectrumDataSetStokes*>(blob) );
     }
#endif
}

} // namespace ampp
} // namespace pelican
