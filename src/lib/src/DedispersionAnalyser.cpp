#include "DedispersionAnalyser.h"


namespace pelican {

namespace lofar {


/**
 *@details DedispersionAnalyser 
 */
DedispersionAnalyser::DedispersionAnalyser( const ConfigNode& config )
    : AbstractModule( config )
{
    
}

/**
 *@details
 */
DedispersionAnalyser::~DedispersionAnalyser()
{
}

AsyncronousJob* DedispersionAnalyser::createJob( DataBlob* data ) {
}

} // namespace lofar
} // namespace pelican
