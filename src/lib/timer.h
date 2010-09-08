#ifndef PELICAN_LOFAR_TIMER_H_
#define PELICAN_LOFAR_TIMER_H_

#include <stdio.h>
#include <stdlib.h>

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

static inline void timerInit(TimerData* data)
{
	data->counter = 0;
	data->timeStart = 0.0;
	data->timeElapsed = 0.0;
	data->timeMin = 0.0;
	data->timeMax = 0.0;
	data->timeAverage = 0.0;
}

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

static inline double timerSec()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + (t.tv_usec * 1.0e-6);
}

static inline void timerStart(TimerData* data)
{
	data->timeStart = timerSec();
	data->timeElapsed = 0.0;
}

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

#endif
