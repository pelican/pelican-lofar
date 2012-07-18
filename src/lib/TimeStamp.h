#ifndef TIMESTAMP_H
#define TIMESTAMP_H
#include <time.h>

/**
 * @file TimeStamp.h
 */

namespace pelican {

namespace lofar {

/**
 * @class TimeStamp
 *  
 * @brief
 *     A class to record timestamps and sccess the time in
 *     various astro formats
 * @details
 * 
 */

class TimeStamp
{
    public:
        TimeStamp( time_t time = 0.0 );
        ~TimeStamp();
        inline double timestamp() const { return _time; };
        double mjd() const;
        static double mjdEpoch();

    private:
        time_t _time;
        mutable time_t _mjd;
        static time_t _mjdEpoch;
};

} // namespace lofar
} // namespace pelican
#endif // TIMESTAMP_H 
