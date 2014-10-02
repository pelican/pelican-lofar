#include "ABPipeline.h"
#include "ABProc.h"
#include "ABData.h"
#include <iostream>

using namespace pelican;
using namespace ampp;

// The constructor. It is good practice to initialise any pointer
// members to zero.
ABPipeline::ABPipeline()
    : AbstractPipeline(), amplifier(0), outputData(0), counter(0)
{
    _dedispersionModule = 0;
    _dedispersionAnalyser = 0;
    _rfiClipper = 0;
}

// The destructor must clean up and created modules and
// any local DataBlob's created.
ABPipeline::~ABPipeline()
{
    delete _dedispersionModule;
    delete _dedispersionAnalyser;
    delete _rfiClipper;
}

// Initialises the pipeline, creating required modules and data blobs,
// and requesting remote data.
void ABPipeline::init()
{
    // Create the pipeline modules and any local data blobs.
    _rfiClipper = (RFI_Clipper *) createModule("RFI_Clipper");
    _dedispersionModule = (DedispersionModule*) createModule("DedispersionModule");
    _dedispersionAnalyser = (DedispersionAnalyser*) createModule("DedispersionAnalyser");
    _dedispersionModule->connect( boost::bind( &DedispersionPipeline::dedispersionAnalysis, this, _1 ) );
    _dedispersionModule->unlockCallback( boost::bind( &DedispersionPipeline::updateBufferLock, this, _1 ) );

    _weightedIntStokes = (WeightedSpectrumDataSet*) createBlob("WeightedSpectrumDataSet");

    // Request remote data.
    requestRemoteData("ABData");
}

// Defines a single iteration of the pipeline.
void ABPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointers to the remote data blob(s) from the supplied hash.
    SpectrumDataSetStokes* stokes = (SpectrumDataSetStokes*) remoteData["ABData"];

    _weightedIntStokes->reset(stokes);
    _rfiClipper_>run(_weightedIntStokes);
    _dedispersionModule->dedisperse(_weightedIntStokes);

    if (counter%10 == 0)
        std::cout << counter << " Chunks processed." << std::endl;

    counter++;
}

