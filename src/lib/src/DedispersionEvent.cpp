#include "DedispersionEvent.h"
#include "DedispersionSpectra.h"


namespace pelican {

namespace lofar {


/**
 *@details DedispersionEvent 
 */
DedispersionEvent::DedispersionEvent( unsigned dmIndex, unsigned timeIndex, const DedispersionSpectra* d )
      : _dm(dmIndex), _time(timeIndex), _data(d)
{
}

/**
 *@details
 */
DedispersionEvent::~DedispersionEvent()
{
}

unsigned DedispersionEvent::timeBin() const
{
   return _time;
}

float DedispersionEvent::dm() const
{
      return  _data->dm( _dm ); 
}

float DedispersionEvent::amplitude() const
{
    return _data->dmAmplitude( _time, _dm );
}

} // namespace lofar
} // namespace pelican
