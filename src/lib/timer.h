#ifndef PELICAN_LOFAR_TIMER_H_
#define PELICAN_LOFAR_TIMER_H_

#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <sys/time.h>
#include "TimerData.h"

namespace pelican {
namespace lofar {

#ifdef TIMING_ENABLED
static inline void timerReport(TimerData* data, const char* message)
{
    data->report(message);
}
#else
#define timerReport(TimerData, char)
#endif

static inline double timerSec()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + (t.tv_usec * 1.0e-6);
}

#ifdef TIMING_ENABLED
#define DEFINE_TIMER(t) TimerData t;
#else
#define DEFINE_TIMER(t) 
#endif

#ifdef TIMING_ENABLED
static inline void timerStart(TimerData* data)
{
    data->timeStart = timerSec();
    data->timeElapsed = 0.0;
}
#else
#define timerStart(TimerData) 
#endif

#ifdef TIMING_ENABLED
static inline void timerUpdate(TimerData* data)
{
    double elapsed = timerSec() - data->timeStart;
    data->timeElapsed = elapsed;
    if (elapsed < data->timeMin) data->timeMin = elapsed;
    if (elapsed > data->timeMax) data->timeMax = elapsed;
    int counter = data->counter;
    data->timeAverage = (elapsed + counter * data->timeAverage) / (counter + 1);
    data->counter++;
    data->timeLast = elapsed;
}
#else
#define timerUpdate(TimerData)
#endif

} // end pelican-lofar namespace
} // end pelican
#endif
