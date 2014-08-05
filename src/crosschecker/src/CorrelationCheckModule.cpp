#include "CorrelationCheckModule.h"


namespace pelican {

namespace ampp {


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

void CorrelationCheckModule::run(const QMap<QString, RTMS_Data>& /*map*/)
{
    // TODO
}

} // namespace ampp
} // namespace pelican
