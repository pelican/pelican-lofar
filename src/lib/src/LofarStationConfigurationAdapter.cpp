#include "LofarStationConfigurationAdapter.h"


namespace pelican {

namespace ampp {


/**
 *@details LofarStationConfigurationAdapter
 */
LofarStationConfigurationAdapter::LofarStationConfigurationAdapter( const ConfigNode& config )
    : AbstractServiceAdapter( config )
{
}

/**
 *@details
 */
LofarStationConfigurationAdapter::~LofarStationConfigurationAdapter()
{
}

/**
 *@details
 */
void LofarStationConfigurationAdapter::deserialise(QIODevice*)
{
}

} // namespace ampp
} // namespace pelican
