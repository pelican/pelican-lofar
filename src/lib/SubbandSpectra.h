#ifndef SUBBAND_SPECTRA_H_
#define SUBBAND_SPECTRA_H_

/**
 * @file SubbandSpectra.h
 */

#include "pelican/data/DataBlob.h"

//#include "LofarDataCube.h"
#include "Spectrum.h"

#include "SubbandTimeSeries.h"

#include <QtCore/QIODevice>
#include <QtCore/QSysInfo>

#include <vector>
#include <complex>
#include <iostream>

namespace pelican {
namespace lofar {

/**
 * @class SubbandSpectra
 *
 * @brief
 * Container class to hold a buffer of blocks of spectra ordered by time,
 * sub-band and polarisation.
 *
 * @details
 * Data is arranged as a Cube of time series objects (a container class
 * encapsulating a time series vector) ordered by:
 *
 *  - time-block (slowest varying dimension)
 *  - sub-band
 *  - polarisation (fastest varying dimension)
 *
 *  The time block dimension is provided for the convenience of being able to
 *  call the PPF channeliser once to generate a number of spectra from each
 *  sub-band and polarisation.
 *
 *
 * @details
 */

template <class T>
class SubbandSpectra : public DataBlob
{
    public:
        /// Constructs an empty sub-band spectra data blob.
        SubbandSpectra(const QString& type = "SubbandSpectra")
        : DataBlob(type), _nTimeBlocks(0), _nSubbands(0), _nPolarisations(0),
          _blockRate(0), _lofarTimestamp(0) {}

        /// Destroys the object.
        virtual ~SubbandSpectra() {}

    public:
        /// Clears the data.
        void clear();

        /// Assign memory for the time stream data blob.
        void resize(unsigned nTimeBlocks, unsigned nSubbands,
                unsigned nPolarisations);

        void resize(unsigned nTimeBlocks, unsigned nSubbands,
                unsigned nPolarisations, unsigned nChannels);

        void resize(unsigned nTimeBlocks, unsigned nSubbands,
                unsigned nPolarisations, unsigned nChannels, T value);

    public:
        /// Returns the number of entries in the data blob.
        unsigned nSpectra() const { return _data.size(); }

        /// Returns the number of blocks of sub-band spectra.
        unsigned nTimeBlocks() const { return _nTimeBlocks; }

        /// Returns the number of sub-bands in the data.
        unsigned nSubbands() const { return _nSubbands; }

        /// Returns the number of polarisations in the data.
        unsigned nPolarisations() const { return _nPolarisations; }

        /// Return the number of channels for the spectrum at time block
        /// \p b, sub-band \p s and polarisation \p.
        unsigned nChannels(unsigned b, unsigned s, unsigned p) const
        { return ptr(b, s, p)->nChannels(); }

        /// Return the block rate (time-span of the entire chunk)
        long getBlockRate() const { return _blockRate; }

        /// Return the block rate (time-span of the entire chunk)
        void setBlockRate(long blockRate) { _blockRate = blockRate; }

        // Return the lofar time-stamp
        long long getLofarTimestamp() const { return _lofarTimestamp; }

        // Set the lofar time-stamp
        void setLofarTimestamp(long long timestamp) { _lofarTimestamp = timestamp; }

        /// Returns a spectrum pointer at index \p i.
        Spectrum<T> * ptr(unsigned i)
        { return (_data.size() > 0 && i < _data.size()) ? &_data[i] : 0; }

        /// Returns a spectrum pointer at index \p i. (const overload).
        Spectrum<T> const * ptr(unsigned i) const
        { return (_data.size() > 0 && i < _data.size()) ? &_data[i] : 0; }

        /// Returns a pointer to the spectrum data for the specified time block
        /// \p b, sub-band \p s, and polarisation \p p.
        Spectrum<T> * ptr(unsigned b, unsigned s, unsigned p)
        {
            // Check the specified index exists.
            if (b >= _nTimeBlocks || s >= _nSubbands || p >= _nPolarisations)
                return 0;
            unsigned i = _index(b, s, p);
            return (_data.size() > 0 && i < _data.size()) ? &_data[i] : 0;
        }

        /// Returns a pointer to the spectrum data for the specified time block
        /// \p b, sub-band \p s, and polarisation \p p (const overload).
        Spectrum<T> const * ptr(unsigned b, unsigned s, unsigned p) const
        {
            // Check the specified index exists.
            if (b >= _nTimeBlocks || s >= _nSubbands || p >= _nPolarisations)
                return 0;
            unsigned idx = _index(b, s, p);
            return (_data.size() > 0 && idx < _data.size()) ? &_data[idx] : 0;
        }

        /// Returns a pointer to the spectrum data for the specified time block
        /// \p b, sub-band \p s, and polarisation \p p (const overload).
        T * spectrum(unsigned b, unsigned s, unsigned p)
        {
            return ptr(b, s, p)->ptr();
        }

        /// Returns a pointer to the spectrum data for the specified time block
        /// \p b, sub-band \p s, and polarisation \p p (const overload).
        T const * spectrum(unsigned b, unsigned s, unsigned p) const
        {
            return ptr(b, s, p)->ptr();
        }

    private:
        /// Returns the data index for a given time block \b, sub-band \s and
        /// polarisation.
        unsigned long _index(unsigned b, unsigned s, unsigned p) const;

    protected:
        std::vector<Spectrum<T> > _data;

        unsigned _nTimeBlocks;
        unsigned _nSubbands;
        unsigned _nPolarisations;
        long     _blockRate;
        long long _lofarTimestamp;
};






// -----------------------------------------------------------------------------
// Inline method/function definitions.
//
template <typename T>
inline void SubbandSpectra<T>::clear()
{
    _data.clear();
    _nTimeBlocks = _nSubbands = _nPolarisations = 0;
    _blockRate = 0;
    _lofarTimestamp = 0;
}


template <typename T>
inline void SubbandSpectra<T>::resize(unsigned nTimeBlocks, unsigned nSubbands,
        unsigned nPolarisations)
{
    _nTimeBlocks = nTimeBlocks;
    _nSubbands = nSubbands;
    _nPolarisations = nPolarisations;
    _data.resize(_nTimeBlocks * _nSubbands * _nPolarisations);
}



template <typename T>
inline void SubbandSpectra<T>::resize(unsigned nTimeBlocks, unsigned nSubbands,
        unsigned nPolarisations, unsigned nChannels)
{
    resize(nTimeBlocks, nSubbands, nPolarisations);
    for (unsigned i = 0; i < _data.size(); ++i) _data[i].resize(nChannels);
}


template <typename T>
inline void SubbandSpectra<T>::resize(unsigned nTimeBlocks, unsigned nSubbands,
        unsigned nPolarisations, unsigned nChannels, T value)
{
    resize(nTimeBlocks, nSubbands, nPolarisations);
    T* s = 0;
    for (unsigned i = 0; i < _data.size(); ++i) {
        _data[i].resize(nChannels);
        s = _data[i].ptr();
        for (unsigned c = 0; c < nChannels; ++c) s[c] = value;
    }
}


template <typename T>
inline unsigned long SubbandSpectra<T>::_index(unsigned b, unsigned s,
        unsigned p) const
{
    return _nPolarisations * (b * _nSubbands + s) + p;
}


// -----------------------------------------------------------------------------
// Template specialisation.
//




/**
 * @class SubbandSpectraC32
 *
 * @brief
 * Data blob to hold a buffer of sub-band spectra in single precision complex
 * format.
 *
 * @details
 * Inherits from the SubbandSpectra template class.
 */

class SubbandSpectraC32 : public SubbandSpectra<std::complex<float> >
{
    public:
        /// Constructor.
        SubbandSpectraC32()
        : SubbandSpectra<std::complex<float> >("SubbandSpectraC32") {}

        /// Destructor.
        ~SubbandSpectraC32() {}

    public:
        /// Write the spectrum to file.
        void write(const QString& fileName) const;

        /// Returns the number of serialised bytes.
        quint64 serialisedBytes() const;

        /// Serialises the data blob.
        void serialise(QIODevice&) const;

        /// Deserialises the data blob.
        void deserialise(QIODevice&, QSysInfo::Endian);
};



/**
 * @class
 * @brief
 * @details
 */

class SubbandSpectraStokes : public SubbandSpectra<float>
{
    public:
        /// Constructor.
        SubbandSpectraStokes()
        : SubbandSpectra<float>("SubbandSpectraStokes") {}

        /// Destructor.
        ~SubbandSpectraStokes() {}

    public:
        quint64 serialisedBytes() const;

        /// Serialises the data blob.
        void serialise(QIODevice&) const;

        /// Deserialises the data blob.
        void deserialise(QIODevice&, QSysInfo::Endian);
};



PELICAN_DECLARE_DATABLOB(SubbandSpectraC32)
PELICAN_DECLARE_DATABLOB(SubbandSpectraStokes)


}// namespace lofar
}// namespace pelican

#endif // SUBBAND_SPECTRA_H_
