#ifndef TIME_SERIES_H_
#define TIME_SERIES_H_

/**
 * @file TimeSeries.h
 */

#include <vector>
#include <complex>

using std::complex;

namespace pelican {
namespace lofar {


/**
 * @class TimeSeries
 *
 * @brief
 *
 * @details
 */

template <class T> class TimeSeries
{
    public:
        /// Constructs an empty time series.
        TimeSeries() : _startTime(0.0), _timeIncrement(0.0) {}

        /// Constructs and assigns memory for the time series.
        TimeSeries(unsigned nTimes)
        : _startTime(0.0), _timeIncrement(0.0)
        {
            resize(nTimes);
        }

        /// Destroys the time stream data blob.
        virtual ~TimeSeries() {}

    public:
        /// Clears the time stream data.
        void clear()
        {
            _times.clear();
            _startTime = 0.0;
            _timeIncrement = 0.0;
        }

        /// Assign memory for the time stream data blob.
        void resize(unsigned nTimes)
        {
            _times.resize(nTimes);
        }

    public:
        /// Returns the number of time samples.
        unsigned nTimes() const { return _times.size(); }

        /// Returns the start time of the data.
        double startTime() const { return _startTime; }

        /// Sets the start time of the data.
        void setStartTime(double value) { _startTime = value; }

        /// Returns the sample delta.
        double timeIncrement() const { return _timeIncrement; }

        /// Sets the time interval between samples.
        void setTimeIncrement(double value) { _timeIncrement = value; }

        /// Returns a pointer to the time stream data.
        T* data() { return _times.size() > 0 ? &_times[0] : 0; }

        /// Returns a pointer to the time stream data (const overload).
        const T* data() const  { return _times.size() > 0 ? &_times[0] : 0; }

        /// To be deprecated soon (dont use!)
        T* ptr()  { return _times.size() > 0 ? &_times[0] : 0; }

        /// To be deprecated soon (dont use!)
        const T* ptr() const  { return _times.size() > 0 ? &_times[0] : 0; }

    protected:
        std::vector<T> _times;
        double _startTime;
        double _timeIncrement;
};


}// namespace lofar
}// namespace pelican

#endif // TIME_SERIES_H_
