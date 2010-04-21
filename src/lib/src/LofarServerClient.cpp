#include "LofarServerClient.h"

namespace pelican {
namespace lofar {

/**
 *@details LofarServerClient
 */
LofarServerClient::LofarServerClient(const ConfigNode& config)
    : PelicanServerClient(config)
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
