#include "LofarStreamDataClient.h"
#include "LofarChunker.h"


#include "utility/memCheck.h"

namespace pelicanLofar {

PELICAN_DECLARE_CHUNKER(LofarChunker);

/**
 *@details LofarStreamDataClient 
 */
LofarStreamDataClient::LofarStreamDataClient()
    : AbstractStreamDataClient()
{
    setChunker( "LofarChunker" );
}

/**
 *@details
 */
LofarStreamDataClient::~LofarStreamDataClient()
{
}

} // namespace pelicanLofar
