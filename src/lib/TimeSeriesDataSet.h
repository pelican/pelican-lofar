#ifndef TIME_SERIES_DATA_SET_H_
#define TIME_SERIES_DATA_SET_H_

/**
 * @file SubbandTimeSeries.h
 */

#include "pelican/data/DataBlob.h"
#include "TimeSeries.h"

#include <vector>
#include <complex>

namespace pelican {
namespace lofar {

/**
 * @class TimeSeriesDataSet
 *
 * @brief
 * Container class to hold a buffer of blocks of time samples ordered by
 * sub-band and polarisation.
 *
 * @details
 * Data is arranged as a Cube of time series objects (a container class
 * encapsulating a time series vector) ordered by:
 *
 * 	1. time block   (Slowest varying dimension)
 *  2. sub-band
 *  3. polarisation (Fastest varying dimension)
 *
 * The time block dimension is provided so that the PPF channeliser
 * which uses this container can generate a number of spectra (=nTimeBlocks)
 * for a given polarisation and sub-band.
 */

template <class T>
class TimeSeriesDataSet : public DataBlob
{
    public:
        /// Constructs an empty time stream data blob.
        TimeSeriesDataSet(const QString& type = "TimeSeriesDataSet")
        : DataBlob(type), _nTimeBlocks(0), _nSubbands(0), _nPolarisations(0),
          _blockRate(0), _lofarTimestamp(0) {}

        /// Destroys the time stream data blob.
        virtual ~TimeSeriesDataSet() {}

    public:
        /// Clears the time stream data.
        void clear();

        /// Assign memory for the cube of time series each of length nTimes.
        void resize(unsigned nTimeBlocks, unsigned nSubbands, unsigned nPols);

        void resize(unsigned nTimeBlocks, unsigned nSubbands, unsigned nPols,
                        unsigned nTimes);

    public:
        /// Returns the number of entries in the data blob.
        unsigned size() const { return _data.size(); }

        /// Returns the number of blocks of sub-band spectra.
        unsigned nTimeBlocks() const { return _nTimeBlocks; }

        /// Returns the number of sub-bands in the data.
        unsigned nSubbands() const { return _nSubbands; }

        /// Returns the number of polarisations in the data.
        unsigned nPolarisations() const { return _nPolarisations; }

        /// Return the number of times for the time series
        /// at time block \p b, sub-band \p s and polarisation \p p.
        unsigned nTimes(unsigned b, unsigned s, unsigned p) const
        { return ptr(b, s, p) ? ptr(b, s, p)->nTimes() : 0; }

        unsigned nTimes(unsigned i = 0) const
        { return _data.size() > 0 && i < _data.size() ? timeSeries(i)->nTimes() : 0; }

        /// Return the block rate (time-span of the entire chunk)
        long getBlockRate() const { return _blockRate; }

        /// Return the block rate (time-span of the entire chunk)
        void setBlockRate(long blockRate) { _blockRate = blockRate; }

        /// Return the lofar time-stamp.
        long long getLofarTimestamp() const { return _lofarTimestamp; }

        /// Set the lofar time-stamp.
        void setLofarTimestamp(long long timestamp) { _lofarTimestamp = timestamp; }

        /// Returns the time series object pointer for the specified time
        /// block \p b, sub-band \p s, and polarisation \p p.
        TimeSeries<T> * timeSeries(unsigned i)
        { return (_data.size() > 0 && i < _data.size()) ? &_data[i] : 0; }

        /// Returns the time series object pointer for the specified time
        /// block \p b, sub-band \p s, and polarisation \p p. (const overload).
        TimeSeries<T> const * timeSeries(unsigned i) const
        { return (_data.size() > 0 && i < _data.size()) ? &_data[i] : 0; }

        /// Returns the time series object pointer for the specified time
        /// block \p b, sub-band \p s, and polarisation \p p.
        TimeSeries<T> * timeSeries(unsigned b, unsigned s, unsigned p)
        { return ptr(b, s, b); }

        /// Returns the time series object pointer for the specified time
        /// block \p b, sub-band \p s, and polarisation \p p. (const overload).
        TimeSeries<T> const * timeSeries(unsigned b, unsigned s, unsigned p) const
        { return ptr(b, s, b); }

        /// Returns a pointer to start of the time series for the specified
        /// time block \p b, sub-band \p s, and polarisation \p p.
        T * timeSeriesData(unsigned b, unsigned s, unsigned p)
        { return ptr(b, s, p)->data(); }

        /// Returns a pointer to start of the time series for the specified
        /// time block \p b, sub-band \p s, and polarisation \p p.
        T const * timeSeriesData(unsigned b, unsigned s, unsigned p) const
        { return ptr(b, s, p)->data(); }

    protected:
        /// *********** DO NOT USE ************
        /// Returns a pointer to the time series data for the specified time block
        /// \p b, sub-band \p s, and polarisation \p p.
        /// *********** DO NOT USE ************
        TimeSeries<T> * ptr(unsigned b, unsigned s, unsigned p)
        {
            if (b >= _nTimeBlocks || s >= _nSubbands || p >= _nPolarisations)
                return 0;
            unsigned i = _index(b, s, p);
            return (_data.size() > 0 && i < _data.size()) ? &_data[i] : 0;
        }

        /// *********** DO NOT USE ************
        /// Returns a pointer to the time series data for the specified
        /// time block \p b, sub-band \p s, and polarisation \p p.
        /// *********** DO NOT USE ************
        const TimeSeries<T> * ptr(unsigned b, unsigned s, unsigned p) const
        {
            if (b >= _nTimeBlocks || s >= _nSubbands || p >= _nPolarisations)
                return 0;
            unsigned i = _index(b, s, p);
            return (_data.size() > 0 && i < _data.size()) ? &_data[i] : 0;
        }

    private:
        /// Data index for a given time block \b, sub-band \s and polarisation.
        unsigned long _index(unsigned b, unsigned s, unsigned p) const;

    private:
        std::vector<TimeSeries<T> > _data;
        unsigned _nTimeBlocks;
        unsigned _nSubbands;
        unsigned _nPolarisations;
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
    _nTimeBlocks = _nSubbands = _nPolarisations = 0;
    _blockRate = 0;
    _lofarTimestamp = 0;
}

template <typename T>
inline void TimeSeriesDataSet<T>::resize(unsigned nTimeBlocks,
        unsigned nSubbands, unsigned nPols)
{
    _nTimeBlocks = nTimeBlocks;
    _nSubbands = nSubbands;
    _nPolarisations = nPols;
    _data.resize(_nTimeBlocks * _nSubbands * _nPolarisations);
}



template <typename T>
inline void TimeSeriesDataSet<T>::resize(unsigned nTimeBlocks,
        unsigned nSubbands, unsigned nPols, unsigned nTimes)
{
    resize(nTimeBlocks, nSubbands, nPols);
    if (nTimes != this->nTimes(0))
        for (unsigned i = 0; i < _data.size(); ++i) _data[i].resize(nTimes);
}


template <typename T>
inline unsigned long TimeSeriesDataSet<T>::_index(unsigned b, unsigned s,
        unsigned p) const
{
    return _nPolarisations * (b * _nSubbands + s) + p;
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
