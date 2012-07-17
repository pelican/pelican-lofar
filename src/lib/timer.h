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

#ifdef TIMING_ENABLED
#define DEFINE_TIMER(t) TimerData t;
#else
#define DEFINE_TIMER(t) 
#endif

#ifdef TIMING_ENABLED
static inline void timerStart(TimerData* data)
{
    data->tick();
}
#else
#define timerStart(TimerData)
#endif

#ifdef TIMING_ENABLED
static inline void timerUpdate(TimerData* data)
{
    data->tock();
}
#else
#define timerUpdate(TimerData)
#endif

} // end pelican-lofar namespace
} // end pelican
#endif
