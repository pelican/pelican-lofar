#ifndef TIME_SERIES_DATA_SET_H_
#define TIME_SERIES_DATA_SET_H_

/**
 * @file SubbandTimeSeries.h
 */

#include "pelican/data/DataBlob.h"

#include <vector>
#include <complex>

namespace pelican {
namespace lofar {

/**
 * @class TimeSeriesDataSet
 *
 * @brief
 *
 * @details
 */

template <class T>
class TimeSeriesDataSet : public DataBlob
{
    public:
        /// Constructs an empty time stream data blob.
        TimeSeriesDataSet(const QString& type = "TimeSeriesDataSet")
        : DataBlob(type), _nSubbands(0), _nPolarisations(0),
          _nTimeBlocks(0), _nTimes(0), _blockRate(0), _lofarTimestamp(0) {}

        /// Destroys the time stream data blob.
        virtual ~TimeSeriesDataSet() {}

    public:
        /// Clears the time stream data.
        void clear();

        /// Assign memory
        void resize(unsigned nSubbands, unsigned nPols, unsigned nTimeBlocks,
                unsigned nTimes);

        /// Assign memory
        void resize(unsigned nSubbands, unsigned nPols, unsigned nTimeBlocks,
                unsigned nTimes, T value);

    public:
        /// Returns the number of entries in the data blob.
        unsigned size() const { return _data.size(); }

        /// Returns the number of sub-bands in the data.
        unsigned nSubbands() const { return _nSubbands; }

        /// Returns the number of polarisations in the data.
        unsigned nPolarisations() const { return _nPolarisations; }

        /// Returns the number of blocks of sub-band spectra.
        unsigned nTimeBlocks() const { return _nTimeBlocks; }

        /// Return the number of times for the time series
        /// at time block \p b, sub-band \p s and polarisation \p p.
        unsigned nTimes() const { return _nTimes; }

        /// Return the block rate (time-span of the entire chunk)
        long getBlockRate() const { return _blockRate; }

        /// Return the block rate (time-span of the entire chunk)
        void setBlockRate(long blockRate) { _blockRate = blockRate; }

        /// Return the lofar time-stamp.
        long long getLofarTimestamp() const { return _lofarTimestamp; }

        /// Set the lofar time-stamp.
        void setLofarTimestamp(long long timestamp) { _lofarTimestamp = timestamp; }

        /// Returns a pointer to start of the time series for the specified
        /// time block \p b, sub-band \p s, and polarisation \p p.
        T * timeSeriesData(unsigned s, unsigned p, unsigned b);

        /// Returns a pointer to start of the time series for the specified
        /// time block \p b, sub-band \p s, and polarisation \p p.
        T const * timeSeriesData(unsigned s, unsigned p, unsigned b) const;

    private:
        /// Data index for a given time block \b, sub-band \s and polarisation.
        unsigned long _index(unsigned s, unsigned p, unsigned b) const;

    private:
        std::vector<T> _data;

        unsigned _nSubbands;
        unsigned _nPolarisations;
        unsigned _nTimeBlocks;
        unsigned _nTimes;

        long _blockRate;
        long long _lofarTimestamp;
};




// -----------------------------------------------------------------------------
// Inline method/function definitions.
//

template <typename T>
inline void TimeSeriesDataSet<T>::clear()
{
    _data.clear();
    _nSubbands = _nPolarisations = _nTimeBlocks = _nTimes = 0;
    _blockRate = 0;
    _lofarTimestamp = 0;
}

template <typename T>
inline void TimeSeriesDataSet<T>::resize(unsigned nSubbands,
        unsigned nPols, unsigned nTimeBlocks, unsigned nTimes)
{
    _nSubbands = nSubbands;
    _nPolarisations = nPols;
    _nTimeBlocks = nTimeBlocks;
    _nTimes = nTimes;
    _data.resize(_nTimeBlocks * _nSubbands * _nPolarisations * _nTimes);
}

template <typename T>
inline void TimeSeriesDataSet<T>::resize(unsigned nSubbands,
        unsigned nPols, unsigned nTimeBlocks, unsigned nTimes, T value)
{
    resize(nSubbands, nPols, nTimeBlocks, nTimes);
    for (unsigned i = 0u; i < _data.size(); ++i) _data[i] = value;
}


template <typename T>
inline unsigned long TimeSeriesDataSet<T>::_index(unsigned s, unsigned p,
        unsigned b) const
{
    return _nTimes * (b  + _nTimeBlocks * (p + s * _nPolarisations));
}

template <typename T> inline T *
TimeSeriesDataSet<T>::timeSeriesData(unsigned s, unsigned p, unsigned b)
{
    if (s >= _nSubbands || p >= _nPolarisations || b >= _nTimeBlocks) return 0;
    unsigned i = _index(s, p, b);
    return _data.size() > 0 && i < _data.size() ? &_data[i] : 0;
}

template <typename T> inline T const *
TimeSeriesDataSet<T>::timeSeriesData(unsigned s, unsigned p, unsigned b) const
{
    if (s >= _nSubbands || p >= _nPolarisations || b >= _nTimeBlocks) return 0;
    unsigned i = _index(s, p, b);
    return _data.size() > 0 && i < _data.size() ? &_data[i] : 0;
}



// -----------------------------------------------------------------------------
// Template specialisation.
//

/**
 * @class SubbandTimeSeriesC32
 * @brief
 * Data container holding a complex floating point time series data cube used
 * in the PPF channeliser.
 */

class TimeSeriesDataSetC32 : public TimeSeriesDataSet<std::complex<float> >
{
    public:
        /// Constructs an empty time stream data blob.
        TimeSeriesDataSetC32()
        : TimeSeriesDataSet<std::complex<float> >("TimeSeriesDataSetC32") {}

        /// Destroys the time stream data blob.
        ~TimeSeriesDataSetC32() {}

    public:
        void write(const QString& fileName) const;
};


// Declare the data blob with the pelican the data blob factory.
PELICAN_DECLARE_DATABLOB(TimeSeriesDataSetC32)

}// namespace lofar
}// namespace pelican
#endif // TIME_SERIES_DATA_SET_H_
