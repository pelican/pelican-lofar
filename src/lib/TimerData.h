#ifndef TIMERDATA_H
#define TIMERDATA_H

#include <stdio.h>
#include <string>

/**
 * @file TimerData.h
 */

namespace pelican {

namespace ampp {

/**
 * @class TimerData
 *  
 * @brief
 * 
 * @details
 * 
 */

class TimerData
{
    public:
        TimerData( const std::string name = "" );
        ~TimerData();
        void reset();
        inline void report( const char* message ) const;
        inline void tick();
        inline void tock();

        int counter;
        double timeStart;
        double threadStart;
        double timeElapsed;
        double threadElapsed;
        double timeMin;
        double timeMax;
        double timeLast;
        double timeAverage;

    private:
        inline double _timerSec( clockid_t type );
        struct timespec _tp; // raw clock
        std::string _name;
};

void TimerData::report( const char* message ) const {
    printf("--------------------Timer Report--------------------\n");
    printf("-- %s\n", message);
    printf("-- Minimum: %.8f sec\n", timeMin);
    printf("-- Maximum: %.8f sec\n", timeMax);
    printf("-- Average: %.8f sec\n", timeAverage);
    printf("-- Latest : %.8f sec\n", timeElapsed);
    printf("-- Thread : %.8f sec\n", threadElapsed);
    printf("-- Counter: %d\n", counter);
    printf("----------------------------------------------------\n");
}

void TimerData::tick() {
    timeStart = _timerSec(CLOCK_MONOTONIC);
    threadStart = _timerSec(CLOCK_THREAD_CPUTIME_ID);
}

void TimerData::tock() {
    timeElapsed = _timerSec(CLOCK_MONOTONIC) - timeStart;
    threadElapsed = _timerSec(CLOCK_THREAD_CPUTIME_ID) - threadStart;
    if (timeElapsed < timeMin) timeMin = timeElapsed;
    if (timeElapsed > timeMax) timeMax = timeElapsed;
    timeAverage = (timeElapsed + counter * timeAverage) / (counter + 1);
    ++counter;
}

double TimerData::_timerSec( clockid_t type )
{
    clock_gettime(type, &_tp); // linux > 2.6.28 only
    return _tp.tv_sec + (_tp.tv_nsec * 1.0e-9);
}

} // namespace ampp
} // namespace pelican
#endif // TIMERDATA_H 
