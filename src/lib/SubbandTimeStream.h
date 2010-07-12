#ifndef SUBBAND_TIME_STREAM_H_
#define SUBBAND_TIME_STREAM_H_

/**
 * @file SubbandTimeStream.h
 */

#include "pelican/data/DataBlob.h"
#include "TimeSeries.h"

#include <vector>
#include <complex>

using std::complex;

namespace pelican {
namespace lofar {

/**
 * @class SubbandTimeStream
 *
 * @brief
 * Container class to hold a buffer of time series data for a number of
 * subbands, polarisations and time blocks.
 *
 * @details
 * ordered by:
 * 	time-block -> subband -> polarisation
 *
 */

template <class T>
class SubbandTimeStream : public DataBlob
{
    public:

        /// Constructs an empty time stream data blob.
        SubbandTimeStream(const QString& type = "SubbandTimeStream")
        : DataBlob(type), _nTimeBlocks(0), _nSubbands(0), _nPolarisations(0) {}

        /// Destroys the time stream data blob.
        virtual ~SubbandTimeStream() {}

    public:
        /// Clears the time stream data.
        void clear()
        {
            _data.clear();
            _nTimeBlocks = 0;
            _nSubbands = 0;
            _nPolarisations = 0;
        }

        /// Assign memory for the time stream data blob.
        void resize(unsigned nTimeBlocks, unsigned nSubbands,
                unsigned nPolarisations)
        {
            _nTimeBlocks = nTimeBlocks;
            _nSubbands = nSubbands;
            _nPolarisations = nPolarisations;
            _data.resize(_nTimeBlocks * _nSubbands * _nPolarisations);
        }

        /// Assign memory for the time stream data blob.
        void resize(unsigned nTimeBlocks, unsigned nSubbands,
                unsigned nPolarisations, unsigned nTimes)
        {
            _nTimeBlocks = nTimeBlocks;
            _nSubbands = nSubbands;
            _nPolarisations = nPolarisations;
            _data.resize(_nTimeBlocks * _nSubbands * _nPolarisations);
            for (unsigned i = 0; i < _data.size(); ++i) {
                _data[i].resize(nTimes);
            }
        }

        /// Returns the data index for a given time block \b, sub-band \s and
        /// polarisation.
        unsigned index(unsigned b, unsigned s, unsigned p) const
        {
            return _nPolarisations * (b * _nSubbands + s) + p;
        }

    public: // accessor methods
        /// Returns the number of entries in the data blob.
        unsigned nTimeSeries() const { return _data.size(); }

        /// Returns the number of blocks of sub-band spectra.
        unsigned nTimeBlocks() const { return _nTimeBlocks; }

        /// Returns the number of sub-bands in the data.
        unsigned nSubbands() const { return _nSubbands; }

        /// Returns the number of polarisations in the data.
        unsigned nPolarisations() const { return _nPolarisations; }

        /// Returns a pointer to the time stream data.
        TimeSeries<T>* ptr() { return _data.size() > 0 ? &_data[0] : 0; }

        /// Returns a pointer to the time stream data (const overload).
        const TimeSeries<T>* ptr() const
        {
            return _data.size() > 0 ? &_data[0] : 0;
        }

        /// Returns a pointer to the data.
        TimeSeries<T>* ptr(unsigned i) {
            return (_data.size() > 0 && i < _data.size()) ? &_data[i] : 0;
        }

        /// Returns a pointer to the data (const. overload).
        const TimeSeries<T>* ptr(unsigned i) const  {
            return (_data.size() > 0 && i < _data.size()) ? &_data[i] : 0;
        }

        /// Returns a pointer to the time series data for the specified time block
        /// \p b, sub-band \p s, and polarisation \p p.
        TimeSeries<T>* ptr(unsigned b, unsigned s, unsigned p)
        {
            // Check the specified index exists.
            if (b >= _nTimeBlocks || s >= _nSubbands || p >= _nPolarisations) {
                return 0;
            }
            unsigned idx = index(b, s, p);
            return (_data.size() > 0 && idx < _data.size()) ? &_data[idx] : 0;
        }


        /// Returns a pointer to the time series data for the specified time block
        /// \p b, sub-band \p s, and polarisation \p p (const overload).
        const TimeSeries<T>* ptr(unsigned b, unsigned s, unsigned p) const
        {
            // Check the specified index exists.
            if (b >= _nTimeBlocks || s >= _nSubbands || p >= _nPolarisations) {
                return NULL;
            }
            unsigned idx = index(b, s, p);
            return (_data.size() > 0 && idx < _data.size()) ? &_data[idx] : 0;
        }

    protected:
        std::vector<TimeSeries<T> > _data;

        unsigned _nTimeBlocks;
        unsigned _nSubbands;
        unsigned _nPolarisations;
};



/**
 * @class SubbandTimeStreamC32
 *
 * @brief
 *
 * @details
 */

class SubbandTimeStreamC32 : public SubbandTimeStream<std::complex<float> >
{
    public:
        /// Constructs an empty time stream data blob.
        SubbandTimeStreamC32()
        : SubbandTimeStream<std::complex<float> >("SubbandTimeStreamC32") {}

        /// Destroys the time stream data blob.
        ~SubbandTimeStreamC32() {}
};

PELICAN_DECLARE_DATABLOB(SubbandTimeStreamC32)

}// namespace lofar
}// namespace pelican

#endif // SUBBAND_TIME_STREAM_H_
