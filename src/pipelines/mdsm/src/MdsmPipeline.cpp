#include "MdsmPipeline.h"
#include <iostream>

namespace pelican {
namespace lofar {


/**
 * @details MdsmPipeline
 */
MdsmPipeline::MdsmPipeline()
    : AbstractPipeline()
{
}

/**
 * @details
 */
MdsmPipeline::~MdsmPipeline()
{
}

/**
 * @details
 * Initialises the pipeline.
 */
void MdsmPipeline::init()
{
    // Create modules
    channeliser = (ChanneliserPolyphase *) createModule("ChanneliserPolyphase");

    // Create local datablob
    polyphaseCoeff = (PolyphaseCoefficients*) createBlob("PolyphaseCoefficients");

    // Hard-code filename, taps and channels.
    QString coeffFileName = "../../../pipelines/mdsm/data/coeffs_512_1.dat";
    int nTaps = 8;
    int nChannels = 512;
    polyphaseCoeff->load(coeffFileName, nTaps, nChannels);

    // Request remote data
    requestRemoteData("TimeStreamData");
}

/**
 * @details
 * Runs the pipeline.
 */
void MdsmPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointer to the remote TimeStreamData data blob
    TimeStreamData* timeData = (TimeStreamData *) remoteData["TimeStreamData"];

    // Run the polyphase channeliser and output module
    channeliser -> run(timeData, polyphaseCoeff, channelisedData);
    std::cout << "Finished the pipeline\n";

}

} // namespace lofar
} // namespace pelican
