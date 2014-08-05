#ifndef DEDISPERSED_TIME_SERIES_H_
#define DEDISPERSED_TIME_SERIES_H_

#include "pelican/data/DataBlob.h"
#include "DedispersedSeries.h"
#include <QtCore/QIODevice>
#include <QtCore/QSysInfo>
#include <iostream>
#include <vector>

namespace pelican {

namespace ampp {

/**
 * @class DedispersedTimeSeries
 *
 * @brief
 * Container class to hold a buffer of spectra generated from sub-bands.
 *
 * The dimensions of the buffer is determined by:
 *  - number of dms.
 *  - number of samples.
 *
 *  Dimension order (outer -> inner):
 *      DMs -> samples
 *
 * @details
 */

template <class T>
class DedispersedTimeSeries : public DataBlob
{
    public:
        /// Constructs an empty dedispersed time series data blob
        DedispersedTimeSeries(const QString& type = "DedispersedTimeSeries")
        : DataBlob(type), _nDMs(0) {}

        /// Destroys the object.
        virtual ~DedispersedTimeSeries() {}

    public:
        /// Clears the data.
        void clear()
        {
            _nDMs = 0;
        }

        /// Assign memory for the dediserpered time series data blob.
        void resize(unsigned nDMs)
        {
            _nDMs = nDMs;
            _values.resize(_nDMs);
        }

    public: // Accessor methods.

        /// Returns the number of sub-bands in the data.
        unsigned nDMs() const { return _nDMs; }

        /// Return the block rate (timespan of the entire chunk)
        double getBlockRate() const { return _blockRate; }

        /// Return the block rate (timespan of the entire chunk)
        void setBlockRate(double blockRate) { _blockRate = blockRate; }

        // Return the lofar timestamp
        double getLofarTimestamp() const { return _lofarTimestamp; }

        // Set the lofar timestamp
        void setLofarTimestamp(double timestamp) { _lofarTimestamp = timestamp; }

        /// Returns a the value for the specified dm and sample
        DedispersedSeries<T>* samples(unsigned dm)
        {
            if (_values.size() >= dm)
                return &_values[dm];
            else return NULL;
        }

        /// Returns a the value for the specified dm and sample (const overload)
        const DedispersedSeries<T>* samples (unsigned dm) const
        {
            if (_values.size() >= dm)
                return &_values[dm];
            else return NULL;
        }

    protected:
         std::vector<DedispersedSeries<T> > _values;

        unsigned _nDMs;
        double     _blockRate;
        double _lofarTimestamp;
};


/**
 * @class DedispersedTimeSeriesC32
 *
 * @brief
 * Data blob to hold a buffer of dedispersed time series in single precision float
 *
 * @details
 * Inherits from the DedispersedTimeSeries template class.
 */
class DedispersedTimeSeriesF32 : public DedispersedTimeSeries<float>
{
    public:
        /// Constructor.
        DedispersedTimeSeriesF32()
        : DedispersedTimeSeries<float>("DedispersedTimeSeriesF32") {}

        /// Destructor.
        ~DedispersedTimeSeriesF32() {}
};

PELICAN_DECLARE_DATABLOB(DedispersedTimeSeriesF32)


} // namespace ampp
} // namespace pelican
#endif // DEDISPERSED_TIME_SERIES_H_
