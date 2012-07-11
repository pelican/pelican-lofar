#ifndef PELICAN_LOFAR_TIMER_H_
#define PELICAN_LOFAR_TIMER_H_

#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <sys/time.h>

typedef struct
{
    int counter;
    double timeStart;
    double timeElapsed;
    double timeMin;
    double timeMax;
    double timeAverage;
} TimerData;

#ifdef TIMING_ENABLED
static inline void timerInit(TimerData* data)
{
    data->counter = 0;
    data->timeStart = 0.0;
    data->timeElapsed = 0.0;
    data->timeMin = DBL_MAX;
    data->timeMax = -DBL_MAX;
    data->timeAverage = 0.0;
}
#else
#define timerInit(TimerData) 
#endif

#ifdef TIMING_ENABLED
static inline void timerReport(TimerData* data, const char* message)
{
    printf("--------------------Timer Report--------------------\n");
    printf("-- %s\n", message);
    printf("-- Minimum: %.4f sec\n", data->timeMin);
    printf("-- Maximum: %.4f sec\n", data->timeMax);
    printf("-- Average: %.4f sec\n", data->timeAverage);
    printf("-- Counter: %d\n", data->counter);
    printf("----------------------------------------------------\n");
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
}
#else
#define timerUpdate(TimerData)
#endif

#endif
