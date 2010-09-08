#include <cstdio>
#include <cstdlib>

#include <sys/time.h>

double timerSec()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + (t.tv_usec * 1.0e-6);
}
