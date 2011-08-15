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
        : DataBlob(type), _nSubbands(0), _nPolarisations(0), _nTimeBlocks(0),
          _nTimesPerBlock(0), _blockRate(0), _lofarTimestamp(0) {}

        /// Destroys the time stream data blob.
        virtual ~TimeSeriesDataSet() {}

    public:
        /// Clears the time stream data.
        void clear();

        /// Resize the data blob.
        void resize(unsigned nTimeBlocks, unsigned nSubbands, unsigned nPols,
                unsigned nTimes);

    public:
        /// Returns the number of samples in the data blob.
        unsigned size() const { return _data.size(); }

        /// Returns the number of sub-bands in the data.
        unsigned nSubbands() const { return _nSubbands; }

        /// Returns the number of polarisations in the data.
        unsigned nPolarisations() const { return _nPolarisations; }

        /// Returns the number of blocks of sub-band spectra.
        unsigned nTimeBlocks() const { return _nTimeBlocks; }

        /// Returns the number of times per time block in the time series.
        unsigned nTimesPerBlock() const { return _nTimesPerBlock; }

    public:
        /// Return the block rate (time-span of the entire chunk)
        double getBlockRate() const { return _blockRate; }

        /// Return the block rate (time-span of the entire chunk)
        void setBlockRate(double blockRate) { _blockRate = blockRate; }

        /// Return the lofar time-stamp.
        double getLofarTimestamp() const { return _lofarTimestamp; }

        /// Set the lofar time-stamp.
        void setLofarTimestamp(double timestamp) { _lofarTimestamp = timestamp; }

    public:
        /// Returns a pointer to start of the time series for the specified
        /// time block \p b, sub-band \p s, and polarisation \p p.
        T * timeSeriesData(unsigned b, unsigned s, unsigned p)
        { return &_data[_index(s, p, b)]; }

        /// Returns a pointer to start of the time series for the specified
        /// time block \p b, sub-band \p s, and polarisation \p p.
        T const * timeSeriesData(unsigned b, unsigned s, unsigned p) const
        { return &_data[_index(s, p, b)]; }

        /// Returns a pointer to start of the data memory block
        T * data() { return &_data[0]; }
        const T* constData() const { return &_data[0]; }

        /// calculates what the index should be given the block, subband, polarisation
        //  primarily used as an aid to optimisation
        static inline long index( unsigned subband, unsigned numTimesPerBlock,
                   unsigned polarisation, unsigned numPolarisations,
                   unsigned block, unsigned numTimeBlocks);

    private:
        /// Time block index.
        unsigned long _index(unsigned s, unsigned p, unsigned b) const;

    private:
        unsigned _nSubbands;
        unsigned _nPolarisations;
        unsigned _nTimeBlocks;
        unsigned _nTimesPerBlock;

    protected:
        std::vector<T> _data;
        double _blockRate;
        double _lofarTimestamp;
};






// -----------------------------------------------------------------------------
// Inline method/function definitions.
//

template <typename T>
inline void TimeSeriesDataSet<T>::clear()
{
    _data.clear();
    _nTimeBlocks = _nSubbands = _nPolarisations = _nTimesPerBlock = 0;
    _blockRate = 0;
    _lofarTimestamp = 0;
}


template <typename T>
inline void TimeSeriesDataSet<T>::resize(unsigned nTimeBlocks,
        unsigned nSubbands, unsigned nPols, unsigned nTimes)
{
    _nSubbands = nSubbands;
    _nPolarisations = nPols;
    _nTimeBlocks = nTimeBlocks;
    _nTimesPerBlock = nTimes;
    _data.resize(nSubbands * nPols * nTimeBlocks * nTimes);
}

template <typename T>
inline long TimeSeriesDataSet<T>::index(unsigned subband,
        unsigned numTimesPerBlock, unsigned polarisation,
        unsigned numPolarisations, unsigned block, unsigned numTimeBlocks)
{
    return numTimesPerBlock * ( numTimeBlocks *
            ( subband * numPolarisations + polarisation ) + block );
}


template <typename T>
inline unsigned long TimeSeriesDataSet<T>::_index(unsigned s, unsigned p,
        unsigned b) const
{
    // subband (outer) -> pol -> timeBlock -> time (inner)
    return _nTimesPerBlock * ( _nTimeBlocks * (s * _nPolarisations + p) + b);
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

        /// Serialises the data blob.
        virtual void serialise(QIODevice&) const;

        /// Deserialises the data blob.
        virtual void deserialise(QIODevice&, QSysInfo::Endian);

    public:
        void write(const QString& fileName,
                int subband = -1, int pol = -1, int block = -1) const;
};


typedef TimeSeriesDataSetC32 LofarTimeStream1;
typedef TimeSeriesDataSetC32 LofarTimeStream2;

// Declare the data blob with the pelican the data blob factory.
PELICAN_DECLARE_DATABLOB(TimeSeriesDataSetC32)
PELICAN_DECLARE_DATABLOB(LofarTimeStream1)
PELICAN_DECLARE_DATABLOB(LofarTimeStream2)

}// namespace lofar
}// namespace pelican
#endif // TIME_SERIES_DATA_SET_H_
