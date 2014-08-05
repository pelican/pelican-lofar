#include "LofarData.h"


namespace pelican {

namespace ampp {


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

} // namespace ampp
} // namespace pelican
