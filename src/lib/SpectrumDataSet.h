#ifndef SPECTRUM_DATA_SET_H
#define SPECTRUM_DATA_SET_H

/**
 * @file SpectrumDataSet.h
 */

#include "pelican/data/DataBlob.h"

#include <QtCore/QIODevice>
#include <QtCore/QSysInfo>

#include <vector>
#include <complex>
#include <algorithm>
#include <iostream>

namespace pelican {
namespace ampp {

/**
 * @class SpectrumDataSet
 *
 * @brief Container class to hold a buffer of blocks of spectra ordered by time, sub-band and polarisation.
 *
 * @details
 */
class SpectrumDataSetBase : public DataBlob
{
    public:
        SpectrumDataSetBase(const QString& type);
        virtual ~SpectrumDataSetBase() {};

        /// Returns the number of spectra in the data blob.
        inline unsigned nSpectra() const
        { return _nTimeBlocks * _nSubbands * _nPolarisations; }

        /// Returns the number of blocks of sub-band spectra.
        inline unsigned nTimeBlocks() const { return _nTimeBlocks; }

        /// Returns the number of sub-bands in the data.
        inline unsigned nSubbands() const { return _nSubbands; }

        /// Returns the number of polarisations in the data.
        inline unsigned nPolarisations() const { return _nPolarisations; }

        /// Returns the number of polarisations in the data.
        virtual inline unsigned nPolarisationComponents() const { 
            return _nPolarisations; 
        }

        /// Return the number of channels for the spectrum specified by index @p i
        inline unsigned nChannels() const { return _nChannels; }

        /// Return the time (in seconds) of the corresponding timeSlice
        double getTime( unsigned sampleNumber ) const { 
            return _startTimestamp + sampleNumber * _blockRate;
        }

        /// Return the block rate (time-span of the each timeslice)
        double getBlockRate() const { return _blockRate; }

        /// Set the block rate (time-span of each timeslice)
        void setBlockRate(double blockRate) { _blockRate = blockRate; }

        /// Return the lofar time-stamp
        inline double getLofarTimestamp() const { return _startTimestamp; }

        /// Set the lofar time-stamp
        void setLofarTimestamp(double timestamp)
        { _startTimestamp = timestamp; }

        /// calculates what the index should be given the block, subband, polarisation (primarily used as an aid to optimisation).
        static inline long index(unsigned subband, unsigned numSubbands,
                   unsigned polarisation, unsigned numPolarisations,
                   unsigned block, unsigned numChannels);

    protected:
        unsigned _nSubbands;
        unsigned _nPolarisations;
        unsigned _nTimeBlocks;
        unsigned _nChannels;

        double  _blockRate;
        double  _startTimestamp;
};

template <class T>
class SpectrumDataSet : public SpectrumDataSetBase
{
    public:
        /// Constructs an empty sub-band spectra data blob.
        SpectrumDataSet(const QString& type = "SpectrumDataSet")
        : SpectrumDataSetBase(type) {}

        /// Destroys the object.
        virtual ~SpectrumDataSet() {}

    public:
        /// Clear the data.
        void clear();

        /// Initialise the data
        void init( const T& value);

        /// Resizes the spectrum data blob to the specified dimensions.
        void resize(unsigned nTimeBlocks, unsigned nSubbands,
                unsigned nPolarisations, unsigned nChannels);


        void resize(const SpectrumDataSet<T>& data)
        {
            resize(data.nTimeBlocks(), data.nSubbands(), data.nPolarisations(),
                    data.nChannels());
        }

    public:

        /// Return the overall size of the data
        int size() const;

        /// Returns a pointer to the data.
        T * data() { return &_data[0]; }

        /// Returns a pointer to the data (const overload).
        T const * data() const { return &_data[0]; }

        /// return true if the data is equivalent
        bool operator!=( const SpectrumDataSet<T>& data ) const {
            return !( data == *this );
        }
        bool operator==( const SpectrumDataSet<T>& data ) const {
            if( data._nTimeBlocks == _nTimeBlocks &&
                    data._nPolarisations == _nPolarisations &&
                    data._nSubbands == _nSubbands &&
                    data._nChannels == _nChannels ) {
                // check data contents
                for( unsigned i=0; i < _data.size(); ++i ) {
                    if( _data[i] != data._data[i] ) {
                        return false;
                    }
                }
                return true;
            }
            return false;
        }

        /// Return an iterator over the data starting at the beginning
        typename std::vector<T>::const_iterator begin() const { return _data.begin(); }

        /// Return an iterator over the data starting at the end
        typename std::vector<T>::const_iterator end() const { return _data.end(); }

        /// Returns a pointer to the spectrum data at index @p i.
        T * spectrumData(unsigned i)
        { return &_data[i * _nChannels]; }

        /// Returns a pointer to the spectrum data at index @p i (const overload).
        T const * spectrumData(unsigned i) const
        { return &_data[i * _nChannels]; }

        /// Returns a pointer to the spectrum data for the specified time block @p b, sub-band @p s, and polarisation @p p (const overload).
        T * spectrumData(unsigned b, unsigned s, unsigned p)
        { return &_data[_index(s, p, b)]; }

        /// Returns a pointer to the spectrum data for the specified time block @p b, sub-band @p s, and polarisation @p p (const overload).
        T const * spectrumData(unsigned b, unsigned s, unsigned p) const
        { return &_data[_index(s, p, b)]; }


    private:
        /// Returns the data index for a given time block @p b, sub-band @p s and polarisation @p p
        unsigned long _index(unsigned s, unsigned p, unsigned b) const;

    protected:
        std::vector<T> _data;

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
    _startTimestamp = 0;
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
    _nSubbands      = nSubbands;
    _nPolarisations = nPolarisations;
    _nTimeBlocks    = nTimeBlocks;
    _nChannels      = nChannels;

    _data.resize(nSubbands * nPolarisations * nTimeBlocks * nChannels);
}

template <typename T>
inline int SpectrumDataSet<T>::size() const
{
    return _data.size();
}

inline long SpectrumDataSetBase::index( unsigned subband, unsigned numSubbands,
        unsigned polarisation, unsigned numPolarisations, unsigned block,
        unsigned numChannels)
{
    return numChannels * ( numPolarisations * ( numSubbands * block + subband )
            + polarisation );
}

template <typename T>
inline
unsigned long SpectrumDataSet<T>::_index(unsigned s, unsigned p, unsigned b) const
{
    return index(s, _nSubbands, p, _nPolarisations, b, _nChannels );
}





// -----------------------------------------------------------------------------
// Template specialisation.
//


/**
 * @class SubbandSpectraC32
 *
 * @brief Data blob to hold a buffer of sub-band spectra in single precision complex format.
 *
 * @details Inherits from the SubbandSpectra template class.
 *
 */

class SpectrumDataSetC32 : public SpectrumDataSet<std::complex<float> >
{
    public:
        /// Constructor.
        SpectrumDataSetC32()
        : SpectrumDataSet<std::complex<float> >("SpectrumDataSetC32") {}

        /// Destructor.
        virtual ~SpectrumDataSetC32() {}

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

        virtual inline unsigned nPolarisationComponents() const { 
            return _nPolarisations*2;
        }
};



/**
 * @class SpectrumDataSetStokes
 *
 * @brief Data blob to hold a buffer of sub-band spectra in stokes format.
 *
 * @details 
 */

class SpectrumDataSetStokes : public SpectrumDataSet<float>
{
    public:
        /// Constructor.
        SpectrumDataSetStokes()
        : SpectrumDataSet<float>("SpectrumDataSetStokes") {}

        /// Destructor.
        virtual ~SpectrumDataSetStokes() {}

    public:
        quint64 serialisedBytes() const;

        /// Serialises the data blob.
        void serialise(QIODevice&) const;

        /// Deserialises the data blob.
        void deserialise(QIODevice&, QSysInfo::Endian);
	/// Methods to link to the associated raw data (spectrumDataSetC32)
	void setRawData(const SpectrumDataSetC32* raw){ _raw = raw; }
	const SpectrumDataSetC32* getRawData() const {return _raw;}
        
    private:
	const SpectrumDataSetC32* _raw;

};


PELICAN_DECLARE_DATABLOB(SpectrumDataSetC32)
PELICAN_DECLARE_DATABLOB(SpectrumDataSetStokes)


}// namespace ampp
}// namespace pelican

#endif // SPECTRUM_DATA_SET_H_
