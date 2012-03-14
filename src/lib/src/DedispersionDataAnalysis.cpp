#include "DedispersionDataAnalysis.h"
#include <QPair>


namespace pelican {

namespace lofar {


/**
 *@details DedispersionDataAnalysis 
 */
DedispersionDataAnalysis::DedispersionDataAnalysis()
    : DataBlob("DedispersedDataAnalysis")
{
}

/**
 *@details
 */
DedispersionDataAnalysis::~DedispersionDataAnalysis() {
}

void DedispersionDataAnalysis::reset( const DedispersionSpectra* data ) {
    _data = data;
}

int DedispersionDataAnalysis::eventsFound() const {
    return _eventIndex.size();
}

void DedispersionDataAnalysis::addEvent( unsigned dmIndex, unsigned timeIndex ) {
    _eventIndex.append( QPair<unsigned,unsigned>(dmIndex, timeIndex) );
}

} // namespace lofar
} // namespace pelican
