#include "LofarStreamDataClient.h"
#include "LofarChunker.h"

#include "pelican/utility/memCheck.h"

namespace pelican {

namespace lofar {

PELICAN_DECLARE_CLIENT(LofarStreamDataClient)

/**
 *@details LofarStreamDataClient
 */
LofarStreamDataClient::LofarStreamDataClient(const ConfigNode& config)
    : DirectStreamDataClient(config)
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
