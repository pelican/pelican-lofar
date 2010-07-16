#include "LofarStreamDataClientNew.h"
#include "LofarChunker.h"

#include "pelican/utility/memCheck.h"

namespace pelican {

namespace lofar {

/**
 *@details LofarStreamDataClient
 */
LofarStreamDataClientNew::LofarStreamDataClientNew(const ConfigNode& configNode,
        const DataTypes& types, const Config* config)
: DirectStreamDataClient(configNode, types, config)
{
    addChunker("SubbandTimeSeriesC32", "LofarChunker");
}

/**
 *@details
 */
LofarStreamDataClientNew::~LofarStreamDataClientNew()
{
}

} // namespace lofar
} // namespace pelican
