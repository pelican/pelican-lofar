#include "LofarServerClient.h"

namespace pelican {
namespace ampp {

/**
 *@details LofarServerClient
 */
LofarServerClient::LofarServerClient(const ConfigNode& configNode,
        const DataTypes& types, const Config* config)
: PelicanServerClient(configNode, types, config)
{
}

/**
 *@details
 */
LofarServerClient::~LofarServerClient()
{
}

} // namespace ampp
} // namespace pelican
