#ifndef SPECTRUM_DATA_SET_H
#define SPECTRUM_DATA_SET_H

/**
 * @file SpectrumDataSet
 */

#include "pelican/data/DataBlob.h"

#include <QtCore/QIODevice>
#include <QtCore/QSysInfo>

#include <vector>
#include <complex>
#include <algorithm>
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
 * WARNING (15/09/2010): this object is in a little bit of flux with respect
 * to the data order. The interface should remain fixed.
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

        /// initialise the data
        void init( const T& value);

        /// Resizes the spectrum data blob to the specified dimensions.
        void resize(unsigned nTimeBlocks, unsigned nSubbands,
                unsigned nPolarisations, unsigned nChannels);
        void resize(const SpectrumDataSet<T>& data ) {
            resize( data.nTimeBlocks(), data.nSubbands(), 
                    data.nPolarisations(), data.nChannels() );
        }

    public:
        /// Returns the number of spectra in the data blob.
        unsigned nSpectra() const
        { return _nTimeBlocks * _nSubbands * _nPolarisations; }

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
        double getBlockRate() const { return _blockRate; }

        /// Return the block rate (time-span of the entire chunk)
        void setBlockRate(double blockRate) { _blockRate = blockRate; }

        /// Return the lofar time-stamp
        double getLofarTimestamp() const
        { return _lofarTimestamp; }

        /// Set the lofar time-stamp
        void setLofarTimestamp(double timestamp)
        { _lofarTimestamp = timestamp; }

        /// return the overall size of the data
        int size() const;

        /// Returns a pointer to the data.
        T * data() { return &_data[0]; }

        /// Returns a pointer to the data (const overload).
        T const * data() const { return &_data[0]; }

        /// Return an iterator over the data starting at the beginning
        typename std::vector<T>::const_iterator begin() const { return _data.begin(); }

        /// Return an iterator over the data starting at the end 
        typename std::vector<T>::const_iterator end() const { return _data.end(); }

        /// Returns a pointer to the spectrum data at index i.
        T * spectrumData(unsigned i)
        { return &_data[i * _nChannels]; }

        /// Returns a pointer to the spectrum data at index i (const overload).
        T const * spectrumData(unsigned i) const
        { return &_data[i * _nChannels]; }

        /// Returns a pointer to the spectrum data for the specified time block
        /// \p b, sub-band \p s, and polarisation \p p (const overload).
        T * spectrumData(unsigned b, unsigned s, unsigned p)
        { return &_data[_index(s, p, b)]; }

        /// Returns a pointer to the spectrum data for the specified time block
        /// \p b, sub-band \p s, and polarisation \p p (const overload).
        T const * spectrumData(unsigned b, unsigned s, unsigned p) const
        { return &_data[_index(s, p, b)]; }

        /// calculates what the index should be given the block, subband, polarisation
        //  primarily used as an aid to optimisation
        static inline long index( unsigned subband, unsigned numSubbands,
                   unsigned polarisation, unsigned numPolarisations,
                   unsigned block, unsigned numChannels );

    private:
        /// Returns the data index for a given time block \b, sub-band \s and
        /// polarisation.
        unsigned long _index(unsigned s, unsigned p, unsigned b) const;

    protected:
        std::vector<T> _data;

    private:
        unsigned _nSubbands;
        unsigned _nPolarisations;
        unsigned _nTimeBlocks;
        unsigned _nChannels;

        double  _blockRate;
        double  _lofarTimestamp;
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
inline void SpectrumDataSet<T>::init(const T& val)
{
    std::fill(_data.begin(),_data.end(), val);
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
inline int SpectrumDataSet<T>::size() const
{
    return _data.size();
}


template <typename T>
inline long SpectrumDataSet<T>::index( unsigned subband, unsigned numSubbands,
                   unsigned polarisation, unsigned numPolarisations,
                   unsigned block, unsigned numChannels
                 )
{
    return numChannels * ( numPolarisations * ( numSubbands * block + subband ) 
                        + polarisation );
}

template <typename T>
inline
unsigned long SpectrumDataSet<T>::_index(unsigned s, unsigned p, unsigned b) const
{
    //  times, polarizations, subbands
    //  return _nChannels * ( _nTimeBlocks * (s * _nPolarisations + p) + b);


    // Polarisation, subbands, times.
    // CHECK: this looks like
    // block [slowest] -> subband -> pol [fastest] (ben - 15/09).
    // NOT what is written above.
    //return _nChannels * ( _nPolarisations * ( _nSubbands * b + s ) + p);
    return index(s, _nSubbands, p, _nPolarisations, b, _nChannels );
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
