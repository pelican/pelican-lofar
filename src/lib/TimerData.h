#ifndef TIMERDATA_H
#define TIMERDATA_H

#include <stdio.h>
#include <string>

/**
 * @file TimerData.h
 */

namespace pelican {

namespace lofar {

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

        int counter;
        double timeStart;
        double timeElapsed;
        double timeMin;
        double timeMax;
        double timeAverage;

    private:
        std::string _name;
};

void TimerData::report( const char* message ) const {
    printf("--------------------Timer Report--------------------\n");
    printf("-- %s\n", message);
    printf("-- Minimum: %.6f sec\n", timeMin);
    printf("-- Maximum: %.6f sec\n", timeMax);
    printf("-- Average: %.6f sec\n", timeAverage);
    printf("-- Counter: %d\n", counter);
    printf("----------------------------------------------------\n");
}

} // namespace lofar
} // namespace pelican
#endif // TIMERDATA_H 
