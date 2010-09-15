#include "RFI_Clipper.h"


namespace pelican {

namespace lofar {


/**
 *@details RFI_Clipper 
 */
RFI_Clipper::RFI_Clipper( const ConfigNode& config )
    : AbstractModule( config )
{
}

/**
 *@details
 */
RFI_Clipper::~RFI_Clipper()
{
}

// Declare this class as a pelican module.
PELICAN_DECLARE_MODULE(RFI_Clipper)

} // namespace lofar
} // namespace pelican
