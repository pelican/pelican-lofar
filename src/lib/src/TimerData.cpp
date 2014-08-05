#include "TimerData.h"
#include <cfloat>


namespace pelican {

namespace ampp {


/**
 *@details TimerData 
 */
TimerData::TimerData( const std::string name ) : _name(name) {
   reset();
}

/**
 *@details
 */
TimerData::~TimerData()
{
     if( _name != "" ) report(_name.c_str());
}

void TimerData::reset() {
    counter = 0;
    timeStart = 0.0;
    timeElapsed = 0.0;
    timeMin = DBL_MAX;
    timeMax = -DBL_MAX;
    timeAverage = 0.0;
}

} // namespace ampp
} // namespace pelican
