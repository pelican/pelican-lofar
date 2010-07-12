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
 * Container class to hold a buffer of time series data.
 *
 * @details
 * Used for time domain processing application such as channeliser modules.
 *
 */

template <class T>
class TimeSeries
{
    public:
        /// Constructs an empty time series.
        TimeSeries() : _nTimes(0), _startTime(0.0), _sampleDelta(0.0) {}

        /// Constructs and assigns memory for the time series.
        TimeSeries(unsigned nTimes)
        : _startTime(0.0), _sampleDelta(0.0)
        {
            resize(nTimes);
        }

        /// Destroys the time stream data blob.
        virtual ~TimeSeries() {}

    public:
        /// Clears the time stream data.
        void clear()
        {
            _data.clear();
            _nTimes = 0;
            _startTime = 0.0;
            _sampleDelta = 0.0;
        }

        /// Assign memory for the time stream data blob.
        void resize(unsigned nTimes)
        {
            _nTimes = nTimes;
            _data.resize(_nTimes);
        }

    public: // accessor methods
        /// Returns the number of entries in the data blob.
        unsigned size() { return _data.size(); }

        /// Returns the number of time samples in the data.
        unsigned nTimes() const { return _nTimes; }

        /// Returns the start time of the data.
        double startTime() const { return _startTime; }

        /// Sets the start time of the data.
        void setStartTime(double value) { _startTime = value; }

        /// Returns the sample delta.
        double sampleDelta() const { return _sampleDelta; }

        /// Sets the time interval between samples.
        void setSampleDelta(double value) { _sampleDelta = value; }

        /// Returns a pointer to the time stream data.
        T* ptr() { return _data.size() > 0 ? &_data[0] : NULL; }

        /// Returns a pointer to the time stream data (const overload).
        const T* ptr() const  { return _data.size() > 0 ? &_data[0] : NULL; }

    protected:
        std::vector<T> _data;
        unsigned _nTimes;
        double _startTime;
        double _sampleDelta;
};


}// namespace lofar
}// namespace pelican

#endif // TIME_SERIES_H_
