#include "DedispersionEvent.h"
#include "DedispersionSpectra.h"


namespace pelican {

namespace ampp {


/**
 *@details DedispersionEvent 
 */
  DedispersionEvent::DedispersionEvent( int dmIndex, unsigned timeIndex, const DedispersionSpectra* d, float mfBinFactor, float mfBinValue )
  : _dm(dmIndex), _time(timeIndex), _data(d), _mfBinFactor(mfBinFactor), _mfBinValue(mfBinValue)
{
}

/**
 *@details
 */
DedispersionEvent::~DedispersionEvent()
{
}

double DedispersionEvent::getTime() const { 
    return _data->getTime( _time );
}

unsigned DedispersionEvent::timeBin() const
{
   return _time;
}

float DedispersionEvent::mfBinning() const
{
   return _mfBinFactor;
}

float DedispersionEvent::mfValue() const
{
   return _mfBinValue;
}

float DedispersionEvent::dm() const
{
    return  _data->dm( _dm ); 
}

float DedispersionEvent::amplitude() const
{
    return _data->dmAmplitude( _time, _dm );
}

} // namespace ampp
} // namespace pelican
