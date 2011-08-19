#include "CorrelationCheckModule.h"


namespace pelican {

namespace lofar {


/**
 *@details CorrelationCheckModule 
 */
CorrelationCheckModule::CorrelationCheckModule(const ConfigNode config)
    : AbstractModule(config)
{
}

/**
 *@details
 */
CorrelationCheckModule::~CorrelationCheckModule()
{
}

void CorrelationCheckModule::run(const QMap<QString, RTMS_Data>& map)
{
    // TODO
}

} // namespace lofar
} // namespace pelican
