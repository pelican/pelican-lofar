#include "LofarStationConfigurationAdapter.h"


namespace pelican {

namespace lofar {


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
void LofarStationConfigurationAdapter::deserialise(QIODevice&)
{
}

} // namespace lofar
} // namespace pelican
