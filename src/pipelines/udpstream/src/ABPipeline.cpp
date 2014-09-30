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
}

// The destructor must clean up and created modules and
// any local DataBlob's created.
ABPipeline::~ABPipeline()
{
    delete amplifier;
    delete outputData;
}

// Initialises the pipeline, creating required modules and data blobs,
// and requesting remote data.
void ABPipeline::init()
{
    // Create the pipeline modules and any local data blobs.
    amplifier = (ABProc*) createModule("ABProc");
    outputData = (ABData*) createBlob("ABData");

    // Request remote data.
    requestRemoteData("ABData");
}

// Defines a single iteration of the pipeline.
void ABPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointers to the remote data blob(s) from the supplied hash.
    ABData* inputData = (ABData*) remoteData["ABData"];

    // Output the input data.
    dataOutput(inputData, "pre");

    // Run each module as required.
    amplifier->run(inputData, outputData);

    // Output the processed data.
    dataOutput(outputData, "post");
    if (counter%10 == 0)
        std::cout << counter << " Chunks processed." << std::endl;

    counter++;
}

