#include "LofarServerClient.h"

namespace pelican {
namespace lofar {

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

} // namespace lofar
} // namespace pelican
