#include "LofarStreamDataClient.h"
#include "LofarChunker.h"

#include "pelican/utility/memCheck.h"

namespace pelican {
namespace lofar {

//PELICAN_DECLARE_CHUNKER(LofarChunker);

/**
 *@details LofarStreamDataClient
 */
LofarStreamDataClient::LofarStreamDataClient(ConfigNode& config, const DataTypes& types)
    : DirectStreamDataClient(config, types)
{
    //setChunker( "LofarChunker" );
}

/**
 *@details
 */
LofarStreamDataClient::~LofarStreamDataClient()
{
}

} // namespace lofar
} // namespace pelican
