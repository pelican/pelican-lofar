#include "LofarStreamDataClient.h"
#include "LofarChunker.h"

#include "pelican/utility/memCheck.h"

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
