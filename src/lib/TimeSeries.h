#ifndef TIME_SERIES_H_
#define TIME_SERIES_H_

/**
 * @file TimeSeries.h
 */

#include "pelican/data/DataBlob.h"
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
 * Multidimensional data holding a time series for a number of sub-bands
 * and polarisations.
 *
 * Data is held in a vector ordered by
 *
 */

template <class T>
class TimeSeries : public DataBlob
{
    public:

        /// Constructs an empty time stream data blob.
        TimeSeries(const QString& type) : DataBlob(type) {
            _nSubbands = 0;
            _nPolarisations = 0;
            _nSamples = 0;
            _startTime = 0.0;
            _sampleDelta = 0.0;
        }

        /// Destroys the time stream data blob.
        virtual ~TimeSeries() {}

    public:
        /// Clears the time stream data.
        void clear()
        {
            _data.clear();
            _nSubbands = 0;
            _nPolarisations = 0;
            _nSamples = 0;
            _startTime = 0.0;
            _sampleDelta = 0.0;
        }

        /// Assign memory for the time stream data blob.
        void resize(unsigned nSubbands, unsigned nPolarisations,
                unsigned nSamples)
        {
            _nSubbands = nSubbands;
            _nPolarisations = nPolarisations;
            _nSamples = nSamples;
            _data.resize(_nSubbands * _nPolarisations * _nSamples);
        }

        /// Returns the data index for a given sub-band, polarisation and
        /// sample.
        unsigned index(unsigned subband, unsigned pol, unsigned sample)
        {
            return _nSamples * ( subband * _nPolarisations + pol) + sample;
        }

    public: // accessor methods
        /// Returns the number of entries in the data blob.
        unsigned size() { return _data.size(); }

        /// Returns the number of sub-bands in the data.
        unsigned nSubbands() const { return _nSubbands; }

        /// Returns the number of polarisations in the data.
        unsigned nPolarisations() const { return _nPolarisations; }

        /// Returns the number of time samples in the data.
        unsigned nSamples() const { return _nSamples; }

        /// Returns the start time of the data.
        double startTime() const { return _startTime; }

        /// Sets the start time of the data.
        void setStartTime(double value) { _startTime = value; }

        /// Returns the sample delta.
        double sampleDelta() const { return _sampleDelta; }

        /// Sets the time interval between samples.
        void setSampleDelta(double value) { _sampleDelta = value; }

        /// Returns a pointer to the time stream data.
        T* data() { return _data.size() > 0 ? &_data[0] : NULL; }

        /// Returns a pointer to the time stream data (const overload).
        const T* data() const  { return _data.size() > 0 ? &_data[0] : NULL; }

        /// Returns a pointer to the time stream data for the specified
        /// /p subband and /p polarisation.
        T* ptr(unsigned subband, unsigned polarisation = 0)
        {
            unsigned index = _nSamples * (subband * _nPolarisations + polarisation);
            return (_data.size() > 0 && subband < _nSubbands
                    && polarisation < _nPolarisations && index < _data.size()) ?
                    &_data[index] : NULL;
        }

        /// Returns a pointer to the time stream data for the specified
        /// /p subband (const overload).
        const T* ptr(unsigned subband, unsigned polarisation = 0) const
        {
            unsigned index = 0;
            return (_data.size() > 0 && subband < _nSubbands
                    && polarisation < _nPolarisations && index < _data.size()) ?
                            &_data[index] : NULL;
        }

    protected:
        std::vector<T> _data;
        unsigned _nSubbands;
        unsigned _nPolarisations;
        unsigned _nSamples;
        double _startTime;
        double _sampleDelta;
};



/**
 * @class TimeStreaData
 *
 * @brief
 * Container class for double format time stream data.
 *
 * @details
 */

class TimeSeriesC32 : public TimeSeries<complex<float> >
{
    public:
        /// Constructs an empty time stream data blob.
        TimeSeriesC32() : TimeSeries<complex<float> >("TimeSeriesC32") {}

        /// Destroys the time stream data blob.
        ~TimeSeriesC32() {}
};

PELICAN_DECLARE_DATABLOB(TimeSeriesC32)

}// namespace lofar
}// namespace pelican

#endif // TIME_SERIES_H_
