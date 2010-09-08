#ifndef PELICAN_LOFAR_TIMER_H_
#define PELICAN_LOFAR_TIMER_H_

#include <cstdio>
#include <cstdlib>

#include <sys/time.h>

static inline double timerSec()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + (t.tv_usec * 1.0e-6);
}


#endif
