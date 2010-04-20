#include "LofarServerClient.h"

namespace pelicanLofar {

/**
 *@details LofarServerClient 
 */
LofarServerClient::LofarServerClient( const ConfigNode& config, const DataTypes& types )
    : PelicanServerClient(config, types)
{
}

/**
 *@details
 */
LofarServerClient::~LofarServerClient()
{
}

} // namespace pelicanLofar
