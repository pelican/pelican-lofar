#include "LofarData.h"


namespace pelican {

namespace lofar {


/**
 *@details LofarData 
 */
LofarData::LofarData( LofarStationConfiguration* config )
    : DataBlob("LofarData"), _config(config)
{
}

/**
 *@details
 */
LofarData::~LofarData()
{
}

const LofarStationConfiguration& LofarData::configuration() const
{
    return *_config;
}

} // namespace lofar
} // namespace pelican
