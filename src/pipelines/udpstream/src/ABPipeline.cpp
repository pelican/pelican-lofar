#include "DedispersionModule.h"
#include "DedispersionDataAnalysis.h"
#include "DedispersionDataAnalysisOutput.h"
#include "WeightedSpectrumDataSet.h"
#include "ABPipeline.h"
#include <boost/bind.hpp>
#include <iostream>

using namespace pelican;
using namespace ampp;

// The constructor. It is good practice to initialise any pointer
// members to zero.
ABPipeline::ABPipeline(const QString& streamIdentifier)
    : AbstractPipeline(), _streamIdentifier(streamIdentifier)
{
    _dedispersionModule = 0;
    _dedispersionAnalyser = 0;
    _rfiClipper = 0;
}

// The destructor must clean up and created modules and
// any local DataBlob's created.
ABPipeline::~ABPipeline()
{
    //delete _dedispersionModule;
    delete _dedispersionAnalyser;
    delete _rfiClipper;
}

// Initialises the pipeline, creating required modules and data blobs,
// and requesting remote data.
void ABPipeline::init()
{
    ConfigNode c = config(QString("ABPipeline"));
    unsigned int history = c.getOption("history", "value", "10").toUInt();
    _minEventsFound = c.getOption("events", "min", "5").toUInt();
    _maxEventsFound = c.getOption("events", "max", "5").toUInt();

    // Create the pipeline modules and any local data blobs.
    _rfiClipper = (RFI_Clipper *) createModule("RFI_Clipper");
    _dedispersionModule = (DedispersionModule*) createModule("DedispersionModule");
    _dedispersionAnalyser = (DedispersionAnalyser*) createModule("DedispersionAnalyser");
    _dedispersionModule->connect( boost::bind( &ABPipeline::dedispersionAnalysis, this, _1 ) );
    _dedispersionModule->unlockCallback( boost::bind( &ABPipeline::updateBufferLock, this, _1 ) );
    _stokesData = createBlobs<SpectrumDataSetStokes>("SpectrumDataSetStokes", history);
    _stokesBuffer = new LockingPtrContainer<SpectrumDataSetStokes>(&_stokesData);
    _weightedIntStokes = (WeightedSpectrumDataSet*) createBlob("WeightedSpectrumDataSet");

    // Request remote data.
    requestRemoteData("SpectrumDataSetStokes");
}

// Defines a single iteration of the pipeline.
void ABPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointers to the remote data blob(s) from the supplied hash.
    SpectrumDataSetStokes* stokes = (SpectrumDataSetStokes*) remoteData["SpectrumDataSetStokes"];
    if( !stokes ) throw(QString("No stokes!"));

    /* to make sure the dedispersion module reads data from a lockable ring
       buffer, copy data to one */
    SpectrumDataSetStokes* stokesBuf = _stokesBuffer->next();
    *stokesBuf = *stokes;

    _weightedIntStokes->reset(stokesBuf);
    _rfiClipper->run(_weightedIntStokes);
    _dedispersionModule->dedisperse(_weightedIntStokes);

    if (0 == counter % 10)
    {
        std::cout << counter << " Chunks processed." << std::endl;
    }

    counter++;
}

void ABPipeline::dedispersionAnalysis( DataBlob* blob ) {
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
              }
            }
        }
        else{
          if (result.eventsFound() >= _minEventsFound && result.eventsFound() <= _maxEventsFound){
            std::cout << "Writing out..." << std::endl;
            dataOutput( &result, "DedispersionDataAnalysis" );
            foreach( const SpectrumDataSetStokes* d, result.data()->inputDataBlobs()) {
              dataOutput( d, "SignalFoundSpectrum" );
            }
          }
        }
      }
}

void ABPipeline::updateBufferLock( const QList<DataBlob*>& freeData ) {
     // find WeightedDataBlobs that can be unlocked
     foreach( DataBlob* blob, freeData ) {
        Q_ASSERT( blob->type() == "SpectrumDataSetStokes" );
        // unlock the pointers to the raw buffer
        _stokesBuffer->unlock( static_cast<SpectrumDataSetStokes*>(blob) );
     }
}

