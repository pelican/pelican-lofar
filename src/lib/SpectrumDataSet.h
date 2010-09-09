#ifndef SPECTRUM_DATA_SET_H
#define SPECTRUM_DATA_SET_H

/**
 * @file SpectrumDataSet
 */

#include "pelican/data/DataBlob.h"

#include "Spectrum.h"

#include <QtCore/QIODevice>
#include <QtCore/QSysInfo>

#include <vector>
#include <complex>
#include <iostream>

namespace pelican {
namespace lofar {

/**
 * @class SpectrumDataSet
 *
 * @brief
 * Container class to hold a buffer of blocks of spectra ordered by time,
 * sub-band and polarisation.
 *
 * @details
 * Data is arranged as a Cube of spectrum objects (a container class
 * encapsulating a spectrum vector) ordered by:
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
class SpectrumDataSet : public DataBlob
{
    public:
        /// Constructs an empty sub-band spectra data blob.
        SpectrumDataSet(const QString& type = "SpectrumDataSet")
        : DataBlob(type), _nSubbands(0), _nPolarisations(0), _nTimeBlocks(0),
          _nChannels(0), _blockRate(0), _lofarTimestamp(0) {}

        /// Destroys the object.
        virtual ~SpectrumDataSet() {}

    public:
        /// Clears the data.
        void clear();

        void resize(unsigned nTimeBlocks, unsigned nSubbands,
                unsigned nPolarisations, unsigned nChannels);

    public:
        /// Returns the number of entries in the data blob.
        unsigned nSpectra() const { return _data.size(); }

        /// Returns the number of blocks of sub-band spectra.
        unsigned nTimeBlocks() const { return _nTimeBlocks; }

        /// Returns the number of sub-bands in the data.
        unsigned nSubbands() const { return _nSubbands; }

        /// Returns the number of polarisations in the data.
        unsigned nPolarisations() const { return _nPolarisations; }

        /// Return the number of channels for the spectrum specified by
        /// index \p i
        unsigned nChannels() const
        { return _nChannels; }

        /// Return the block rate (time-span of the entire chunk)
        long getBlockRate() const { return _blockRate; }

        /// Return the block rate (time-span of the entire chunk)
        void setBlockRate(long blockRate) { _blockRate = blockRate; }

        /// Return the lofar time-stamp
        long long getLofarTimestamp() const { return _lofarTimestamp; }

        /// Set the lofar time-stamp
        void setLofarTimestamp(long long timestamp) { _lofarTimestamp = timestamp; }

//        /// Returns a spectrum pointer at index \p i.
//        Spectrum<T> * spectrum(unsigned i)
//        { return (_data.size() > 0 && i < _data.size()) ? &_data[i] : 0; }
//
//        /// Returns a spectrum pointer at index \p i. (const overload).
//        Spectrum<T> const * spectrum(unsigned i) const
//        { return (_data.size() > 0 && i < _data.size()) ? &_data[i] : 0; }

        T * data() { return &_data[0]; }

        T const * data() const { return &_data[0]; }

        T * spectrumData(unsigned i) {
            return &_data[i * _nChannels];
        }

        T const * spectrumData(unsigned i) const {
            return &_data[i * _nChannels];
        }

        /// Returns a pointer to the spectrum data for the specified time block
        /// \p b, sub-band \p s, and polarisation \p p (const overload).
        T * spectrumData(unsigned b, unsigned s, unsigned p)
        { return &_data[_index(s, p, b)]; }

        /// Returns a pointer to the spectrum data for the specified time block
        /// \p b, sub-band \p s, and polarisation \p p (const overload).
        T const * spectrumData(unsigned b, unsigned s, unsigned p) const
        { return &_data[_index(s, p, b)]; }


    private:
        /// Returns the data index for a given time block \b, sub-band \s and
        /// polarisation.
        unsigned long _index(unsigned s, unsigned p, unsigned b) const;

    private:
        std::vector<T> _data;

        unsigned _nSubbands;
        unsigned _nPolarisations;
        unsigned _nTimeBlocks;
        unsigned _nChannels;

        long     _blockRate;
        long long _lofarTimestamp;
};






// -----------------------------------------------------------------------------
// Inline method/function definitions.
//
template <typename T>
inline void SpectrumDataSet<T>::clear()
{
    _data.clear();
    _nTimeBlocks = _nSubbands = _nPolarisations = _nChannels = 0;
    _blockRate = 0;
    _lofarTimestamp = 0;
}


template <typename T>
inline void SpectrumDataSet<T>::resize(unsigned nTimeBlocks, unsigned nSubbands,
        unsigned nPolarisations, unsigned nChannels)
{
    _nSubbands = nSubbands;
    _nPolarisations = nPolarisations;
    _nTimeBlocks = nTimeBlocks;
    _nChannels = nChannels;
    _data.resize(nSubbands * nPolarisations * nTimeBlocks * nChannels);
}


template <typename T>
inline unsigned long SpectrumDataSet<T>::_index(unsigned s, unsigned p,
        unsigned b) const
{
    // Returns the index for the data with the order:
    // Subband, s (slowest) -> polarisation -> block (fastest).
    return _nChannels * ( _nTimeBlocks * (s * _nPolarisations + p) + b);
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

class SpectrumDataSetC32 : public SpectrumDataSet<std::complex<float> >
{
    public:
        /// Constructor.
        SpectrumDataSetC32()
        : SpectrumDataSet<std::complex<float> >("SpectrumDataSetC32") {}

        /// Destructor.
        ~SpectrumDataSetC32() {}

    public:
        /// Write the spectrum to file.
        void write(const QString& fileName,
                int s = -1, int p = -1, int b = -1) const;

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

class SpectrumDataSetStokes : public SpectrumDataSet<float>
{
    public:
        /// Constructor.
        SpectrumDataSetStokes()
        : SpectrumDataSet<float>("SpectrumDataSetStokes") {}

        /// Destructor.
        ~SpectrumDataSetStokes() {}

    public:
        quint64 serialisedBytes() const;

        /// Serialises the data blob.
        void serialise(QIODevice&) const;

        /// Deserialises the data blob.
        void deserialise(QIODevice&, QSysInfo::Endian);
};


PELICAN_DECLARE_DATABLOB(SpectrumDataSetC32)
PELICAN_DECLARE_DATABLOB(SpectrumDataSetStokes)


}// namespace lofar
}// namespace pelican

#endif // SPECTRUM_DATA_SET_H_
