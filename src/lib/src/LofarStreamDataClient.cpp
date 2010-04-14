#include "LofarStreamDataClient.h"
#include "LofarChunker.h"

#include "pelican/utility/memCheck.h"

namespace pelicanLofar {

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

} // namespace pelicanLofar
