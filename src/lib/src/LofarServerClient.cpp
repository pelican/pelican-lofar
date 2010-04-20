#include "LofarServerClient.h"

namespace pelican {
namespace lofar {

/**
 *@details LofarServerClient
 */
LofarServerClient::LofarServerClient(const ConfigNode& config, const DataTypes& types)
    : PelicanServerClient(config, types)
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
