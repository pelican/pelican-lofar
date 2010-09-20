#ifndef DEDISPERSED_SERIES_H_
#define DEDISPERSED_SERIES_H_

/**
 * @file DedispersedSeries.h
 */

#include <vector>
#include <complex>

using std::complex;

/**
 * @class DedispersedSeries
 * @brief
 * @details
 */
template <class T> class DedispersedSeries
{
    public:
        /// Constructs an empty time series.
        DedispersedSeries() : _startTime(0.0), _timeIncrement(0.0) {}

        /// Constructs and assigns memory for the time series.
        DedispersedSeries(unsigned nSamples)
        : _startTime(0.0), _timeIncrement(0.0), _dmValue(0)
        {
            resize(nSamples);
        }

        /// Destroys the time stream data blob.
        virtual ~DedispersedSeries() {}

    public:
        /// Clears the time stream data.
        void clear()
        {
            _samples.clear();
            _startTime = 0.0;
            _timeIncrement = 0.0;
            _dmValue = 0.0;
        }

        /// Assign memory for the time stream data blob.
        void resize(unsigned nSamples)
        {
            _samples.resize(nSamples);
        }

    public:
        /// Returns the number of time samples.
        unsigned nSamples() const { return _samples.size(); }

        /// Return the dm value associated with this time series
        float dmValue() const { return _dmValue; }

        /// Sets the dm value associated with this time series
        void setDmValue(float value) { _dmValue = value; }

        /// Returns the start time of the data.
        double startTime() const { return _startTime; }

        /// Sets the start time of the data.
        void setStartTime(double value) { _startTime = value; }

        /// Returns the sample delta.
        double timeIncrement() const { return _timeIncrement; }

        /// Sets the time interval between samples.
        void setTimeIncrement(double value) { _timeIncrement = value; }

        /// Returns a pointer to the time stream data.
        T* getData() { return _samples.size() > 0 ? &_samples[0] : 0; }

        /// Returns a pointer to the time stream data (const overload).
        const T* getData() const  { return _samples.size() > 0 ? &_samples[0] : 0; }

        /// To be deprecated soon (dont use!)
        T* ptr()  { return _samples.size() > 0 ? &_samples[0] : 0; }

        /// To be deprecated soon (dont use!)
        const T* ptr() const  { return _samples.size() > 0 ? &_samples[0] : 0; }

    protected:
        std::vector<T> _samples;
        double _startTime;
        double _timeIncrement;
        float  _dmValue;
};

#endif // DEDISPERSED_SERIES_H_
