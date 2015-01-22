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
    ConfigNode c = config(QString("SigprocPipeline"));
    unsigned int history = c.getOption("history", "value", "10").toUInt();
    _minEventsFound = c.getOption("events", "min", "5").toUInt();
    _maxEventsFound = c.getOption("events", "max", "5").toUInt();

    _rfiClipper = (RFI_Clipper *) createModule("RFI_Clipper");
    _dedispersionModule = (DedispersionModule*) createModule("DedispersionModule");
    _dedispersionAnalyser = (DedispersionAnalyser*) createModule("DedispersionAnalyser");
    _dedispersionModule->connect( boost::bind( &SigprocPipeline::dedispersionAnalysis, this, _1 ) );
    _dedispersionModule->unlockCallback( boost::bind( &SigprocPipeline::updateBufferLock, this, _1 ) );
    _stokesData = createBlobs<SpectrumDataSetStokes>("SpectrumDataSetStokes", history);
    _stokesBuffer = new LockingPtrContainer<SpectrumDataSetStokes>(&_stokesData);
    _weightedIntStokes = (WeightedSpectrumDataSet*) createBlob("WeightedSpectrumDataSet");

    _iter = 0;

    // Request remote data
    requestRemoteData("SpectrumDataSetStokes");
}

void SigprocPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    SpectrumDataSetStokes* stokes = (SpectrumDataSetStokes*)remoteData["SpectrumDataSetStokes"];
    if( ! stokes ) throw(QString("no STOKES!"));

    /* to make sure the dedispersion module reads data from a lockable ring
       buffer, copy data to one */
    SpectrumDataSetStokes* stokesBuf = _stokesBuffer->next();
    *stokesBuf = *stokes;

    _weightedIntStokes->reset(stokesBuf);
    _rfiClipper->run(_weightedIntStokes);
    dataOutput(stokesBuf, "SigprocStokesWriter");
    _dedispersionModule->dedisperse(_weightedIntStokes);
    ++_iter;
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
            std::cout << "here!" << std::endl;
            dataOutput( &result, "DedispersionDataAnalysis" );
       //     foreach( const SpectrumDataSetStokes* d, result.data()->inputDataBlobs()) {
       //       dataOutput( d, "SignalFoundSpectrum" );
       //       //                    dataOutput( d->getRawData(), "RawDataFoundSpectrum" );
       //     }
          }
        }
      }
}

void SigprocPipeline::updateBufferLock( const QList<DataBlob*>& freeData ) {
     // find WeightedDataBlobs that can be unlocked
     foreach( DataBlob* blob, freeData ) {
        Q_ASSERT( blob->type() == "SpectrumDataSetStokes" );
        // unlock the pointers to the raw buffer
        // _rawBuffer->unlock( static_cast<SpectrumDataSetStokes*>(blob)->getRawData() );
        _stokesBuffer->unlock( static_cast<SpectrumDataSetStokes*>(blob) );
     }
}

} // namespace ampp
} // namespace pelican
