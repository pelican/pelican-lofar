#include "DedispersionSpectra.h"


namespace pelican {

namespace lofar {


/**
 *@details DedispersionSpectra 
 */
DedispersionSpectra::DedispersionSpectra()
    : DataBlob("DedispersionSpectra"), _timeBins(0)
{
}

/**
 *@details
 */
DedispersionSpectra::~DedispersionSpectra()
{
}

void DedispersionSpectra::resize( unsigned timebins, unsigned dedispersionBins ) { 
    _timeBins = timebins;
    _data.resize( timebins*dedispersionBins ); 
}

} // namespace lofar
} // namespace pelican
