#include "LofarStreamDataClient.h"
#include "LofarChunker.h"

namespace pelican {

namespace lofar {

/**
 *@details LofarStreamDataClient
 */
LofarStreamDataClient::LofarStreamDataClient(const ConfigNode& configNode,
        const DataTypes& types, const Config* config)
: DirectStreamDataClient(configNode, types, config)
{
    addChunker("TimeSeriesDataSetC32", "LofarChunker");
}


/**
 *@details
 */
LofarStreamDataClient::~LofarStreamDataClient()
{
}


} // namespace lofar
} // namespace pelican
