#ifndef SUBBAND_SPECTRA_H_
#define SUBBAND_SPECTRA_H_

/**
 * @file SubbandSpectra.h
 */

#include "pelican/data/DataBlob.h"
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
 * Container class to hold a buffer of spectra generated from sub-bands.
 *
 * The dimensions of the buffer is determined by:
 *  - number of sub-bands.
 *  - number of polarisations for each sub-band.
 *  - number of blocks of sub-band spectra.
 *
 *  Dimension order (outer -> inner):
 *
 *      Time blocks -> sub-bands -> polarisations.
 *
 *  Polarisations is therefore the fastest varying index.
 *
 *  Each buffer entry contains a spectrum data blob.
 *
 * @details
 */

template <class T>
class SubbandSpectra : public DataBlob
{
    public:
        /// Constructs an empty sub-band spectra data blob.
        SubbandSpectra(const QString& type = "SubbandSpectra")
        : DataBlob(type), _nTimeBlocks(0), _nSubbands(0), _nPolarisations(0) {}

        /// Constructs an empty sub-band spectra from a SubbandTimeSeries
        SubbandSpectra(const SubbandTimeSeries<T>& ts, const QString& type = "SubbandSpectra")
        : DataBlob(type)
        {
            resize(ts.nTimeBlocks(),ts.nSubbands(), ts.nPolarisations(), 1 );
            for (unsigned t = 0; t < ts.nTimeBlocks(); ++t) {
                for (unsigned p = 0; p < ts.nPolarisations(); ++p) {
                    for (unsigned s = 0; s < ts.nSubbands(); ++s) {
                        int idx = index( t, s, p);
                        T* data = ts.ptr(t,s,p)->ptr();
                        _spectra[idx].push_back(data[idx]);
                    }
                }
            }
        }

        /// Destroys the object.
        virtual ~SubbandSpectra() {}

    public:
        /// Clears the data.
        void clear()
        {
            _spectra.clear();
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
            _spectra.resize(_nTimeBlocks * _nSubbands * _nPolarisations);
        }

        /// Assign memory for the time stream data blob.
        void resize(unsigned nTimeBlocks, unsigned nSubbands,
                unsigned nPolarisations, unsigned nChannels)
        {
            resize( nTimeBlocks, nSubbands, nPolarisations);
            _nTimeBlocks = nTimeBlocks;
            _nSubbands = nSubbands;
            _nPolarisations = nPolarisations;
            _spectra.resize(_nTimeBlocks * _nSubbands * _nPolarisations);
            for (unsigned i = 0; i < _spectra.size(); ++i) {
                _spectra[i].resize(nChannels);
            }
        }

        /// Returns the data index for a given time block \b, sub-band \s and
        /// polarisation.
        unsigned index(unsigned b, unsigned s, unsigned p) const
        {
            return _nPolarisations * (b * _nSubbands + s) + p;
        }

        /// add the supplied subband spectra values to this object
        SubbandSpectra& operator+(const SubbandSpectra& spectra)
        {
            for (unsigned t = 0; t < spectra.nTimeBlocks(); ++t) {
                for (unsigned p = 0; p < spectra.nPolarisations(); ++p) {
                    for (unsigned s = 0; s < spectra.nSubbands(); ++s) {
                        int idx = index( t, s, p);
                         T* spectrum = spectra->ptr(t, s, p)->ptr();
                         unsigned nChannels = spectra->ptr(t,s,p)->nChannels();
                         for (unsigned i = 0; i < nChannels; ++i) {
                             _spectra[idx][i] += spectrum[i];
                         }
                    }
                }
            }
        }

    public: // Accessor methods.

        /// Returns the number of entries in the data blob.
        unsigned nSpectra() const { return _spectra.size(); }

        /// Returns the number of blocks of sub-band spectra.
        unsigned nTimeBlocks() const { return _nTimeBlocks; }

        /// Returns the number of sub-bands in the data.
        unsigned nSubbands() const { return _nSubbands; }

        /// Returns the number of polarisations in the data.
        unsigned nPolarisations() const { return _nPolarisations; }

        /// Return the block rate (timespan of the entire chunk)
        long getBlockRate() const { return _blockRate; }

        /// Return the block rate (timespan of the entire chunk)
        void setBlockRate(long blockRate) { _blockRate = blockRate; }

        // Return the lofar timestamp
        long long getLofarTimestamp() const { return _lofarTimestamp; }

        // Set the lofar timestamp
        void setLofarTimestamp(long long timestamp) { _lofarTimestamp = timestamp; }

        /// Returns a pointer to the data.
        Spectrum<T>* ptr() { return _spectra.size() > 0 ? &_spectra[0] : NULL; }

        /// Returns a pointer to the data (const overload).
        const Spectrum<T>* ptr() const  {
            return _spectra.size() > 0 ? &_spectra[0] : NULL;
        }

        /// Returns a pointer to the data.
        Spectrum<T>* ptr(unsigned i) {
            return (_spectra.size() > 0 && i < _spectra.size()) ?
                    &_spectra[i] : NULL;
        }

        /// Returns a pointer to the data (const overload).
        const Spectrum<T>* ptr(unsigned i) const  {
            return (_spectra.size() > 0 && i < _spectra.size()) ?
                    &_spectra[i] : NULL;
        }

        /// Returns a pointer to the spectrum data for the specified time block
        /// \p b, sub-band \p s, and polarisation \p p.
        Spectrum<T>* ptr(unsigned b, unsigned s, unsigned p)
        {
            // Check the specified index exists.
            if (b >= _nTimeBlocks || s >= _nSubbands || p >= _nPolarisations) {
                return NULL;
            }
            unsigned idx = index(b, s, p);
            return (_spectra.size() > 0 && idx < _spectra.size()) ?
                            &_spectra[idx] : NULL;
        }

        /// Returns a pointer to the spectrum data for the specified time block
        /// \p b, sub-band \p s, and polarisation \p p (const overload).
        const Spectrum<T>* ptr(unsigned b, unsigned s, unsigned p) const
        {
            // Check the specified index exists.
            if (b >= _nTimeBlocks || s >= _nSubbands || p >= _nPolarisations) {
                return NULL;
            }
            unsigned idx = index(b, s, p);
            return (_spectra.size() > 0 && idx < _spectra.size()) ?
                    &_spectra[idx] : NULL;
        }

    protected:
        std::vector<Spectrum<T> > _spectra;

        unsigned _nTimeBlocks;
        unsigned _nSubbands;
        unsigned _nPolarisations;
        long     _blockRate;
        long long _lofarTimestamp;
};



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
 *
 * @brief
 *
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
