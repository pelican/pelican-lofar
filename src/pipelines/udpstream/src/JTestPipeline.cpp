#include "JTestPipeline.h"
#include "JTestProc.h"
#include "JTestData.h"
#include <iostream>

using namespace pelican;
using namespace ampp;

// The constructor. It is good practice to initialise any pointer
// members to zero.
JTestPipeline::JTestPipeline()
    : AbstractPipeline(), amplifier(0), outputData(0), counter(0)
{
}

// The destructor must clean up and created modules and
// any local DataBlob's created.
JTestPipeline::~JTestPipeline()
{
    delete amplifier;
    delete outputData;
}

// Initialises the pipeline, creating required modules and data blobs,
// and requesting remote data.
void JTestPipeline::init()
{
    // Create the pipeline modules and any local data blobs.
    amplifier = (JTestProc*) createModule("JTestProc");
    outputData = (JTestData*) createBlob("JTestData");

    // Request remote data.
    requestRemoteData("JTestData");
}

// Defines a single iteration of the pipeline.
void JTestPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointers to the remote data blob(s) from the supplied hash.
    JTestData* inputData = (JTestData*) remoteData["JTestData"];

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

