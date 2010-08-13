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
    _iteration = 0;
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
//    // Create modules
//    channeliser = (ChanneliserPolyphase *) createModule("ChanneliserPolyphase");
////     tcpBlobServer = (PelicanTCPBlobServer *) createModule("PelicanTCPBlobServer");
//
//    // Create local datablobs
//    polyphaseCoeff = (PolyphaseCoefficients*) createBlob("PolyphaseCoefficients");
//    channelisedData = (ChannelisedStreamData*) createBlob("ChannelisedStreamData");
//
//    // Hard-code filename, taps and channels.
//    // FIXME These are quick hard-coded hacks at the moment.
//    //QString coeffFileName = "/code/pelican-lofar/release/pipelines/mdsm/data/coeffs_512_1.dat";
//    QString coeffFileName = "/home/bmort/pelican/pelican-lofar/src/intel_release/pipelines/mdsm/data/coeffs_512_1.dat";
//    int nTaps = 8;
//    int nChannels = 512;
//    polyphaseCoeff->load(coeffFileName, nTaps, nChannels);
//
//    // Request remote data
//    requestRemoteData("TimeStreamData");
}

/**
 * @details
 * Runs the pipeline.
 */
void MdsmPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
//    // Get pointer to the remote TimeStreamData data blob
//    TimeStreamData* timeData = (TimeStreamData *) remoteData["TimeStreamData"];
//
//    // Run the polyphase channeliser.
//    channeliser -> run(timeData, polyphaseCoeff, channelisedData);
//
//    // Send the blob using the output module.
////    tcpBlobServer->send("ChannelisedStreamData", channelisedData);
//    dataOutput( channelisedData, "ChannelisedStreamData" );
//
//    if (_iteration % 200 == 0) std::cout << "Finished the MDSM pipeline, iteration " << _iteration << std::endl;
//    _iteration++;
}

} // namespace lofar
} // namespace pelican
