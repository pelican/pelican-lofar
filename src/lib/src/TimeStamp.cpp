#include "TimeStamp.h"


namespace pelican {

namespace lofar {


/**
 *@details TimeStamp 
 */
TimeStamp::TimeStamp( double t )
   : _time(t), _mjd(0.0)
{
}

/**
 *@details
 */
TimeStamp::~TimeStamp()
{
}

double TimeStamp::mjd() const {
    if( _mjd == 0.0 ) 
        _mjd = (_time-mjdEpoch())/86400 + 55562.0;
    return _mjd;
}

double TimeStamp::mjdEpoch() {
    struct tm tm;
    // MJD of 1/1/11 is 55562
    strptime("2011-1-1 0:0:0", "%Y-%m-%d %H:%M:%S", &tm);
    static time_t _mjdEpoch = mktime(&tm);
    return _mjdEpoch;
}

} // namespace lofar
} // namespace pelican
