#ifndef POLYPHASE_COEFFICIENTS_H_
#define POLYPHASE_COEFFICIENTS_H_

#include "pelican/data/DataBlob.h"

#include <QtCore/QIODevice>

#include <boost/multi_array.hpp>

#include <vector>
#include <complex>

namespace pelican {
namespace ampp {


/**
 * @class T_PolyphaseCoefficients
 *
 * @ingroup pelican_lofar
 *
 * @brief
 * Container class to hold polyphase filter coefficients.
 *
 * @details
 */

template <class T>
class T_PolyphaseCoefficients : public DataBlob
{
    public:
        /// Constructs a polyphase coefficients object.
        T_PolyphaseCoefficients(const QString& type)
        : DataBlob(type), _nTaps(0), _nChannels(0)
        {}

        /// Constructs a polyphase coefficients object with the specified
        /// dimensions.
        T_PolyphaseCoefficients(unsigned nTaps, unsigned nChannels,
                const QString& type) : DataBlob(type)
        { resize(nTaps, nChannels); }

        /// Destroys the object.
        virtual ~T_PolyphaseCoefficients() {}

    public:
        /// Clears the coefficients.
        void clear()
        {
            _coeff.clear();
            _nTaps = 0; _nChannels = 0;
        }

        /// Resizes the coefficient vector for nTaps and nChannels.
        void resize(unsigned nTaps, unsigned nChannels)
        {
            _nTaps = nTaps;
            _nChannels = nChannels;
            _coeff.resize(_nTaps * _nChannels);
        }

    public: // Accessor methods.
        /// Returns the number of coefficients.
        unsigned size() const
        { return _coeff.size(); }

        /// Returns the number of filter taps.
        unsigned nTaps() const
        { return _nTaps; }

        /// Returns the number of channels.
        unsigned nChannels() const
        { return _nChannels; }

        /// Returns a pointer to the vector of coefficients.
        T* ptr()
        { return _coeff.size() > 0 ? &_coeff[0] : NULL; }

        /// Returns a pointer to the vector of coefficients (const overload).
        const T* ptr() const
        { return _coeff.size() > 0 ? &_coeff[0] : NULL; }

    protected:
        std::vector<T> _coeff;
        unsigned _nTaps;
        unsigned _nChannels;
};


/**
 * @class PolyphaseCoefficients
 *
 * @brief
 * Container class to hold polyphase filter coefficients.
 *
 * @details
 * Template specialisation for complex double type.
 */

class PolyphaseCoefficients : public T_PolyphaseCoefficients<double>
{
    public:
        friend class PolyphaseCoefficientsTest;

    public:
        typedef enum { HAMMING, BLACKMAN, GAUSSIAN, KAISER } FirWindow;

    public:

        /// Constructs an empty polyphase filter coefficient data blob.
        PolyphaseCoefficients() : T_PolyphaseCoefficients<double>
        ("PolyphaseCoefficients") {}

        /// Constructs a polyphase filter coefficient data blob.
        PolyphaseCoefficients(unsigned nTaps, unsigned nChannels) :
            T_PolyphaseCoefficients<double>(nTaps, nChannels,
                    "PolyphaseCoefficients") {}

        /// Constructs a polyphase filter coefficient data blob loading values
        /// the specified file.
        PolyphaseCoefficients(const QString& fileName, unsigned nTaps,
                unsigned nChannels) :
            T_PolyphaseCoefficients<double>("PolyphaseCoefficients")
        {
            load(fileName, nTaps, nChannels);
        }


    public:
        /// Load coefficients from matlab coefficient dump.
        void load(const QString& fileName, unsigned nFilterTaps,
                unsigned nChannels);

        void genereateFilter(unsigned nTaps, unsigned nChannels,
                FirWindow windowType = KAISER);

    private:
        // The following methods are taken from LOFAR CNProc FIR.cc under GPL
        double _besselI0(double x);

        void _kaiser(int n, double beta, double* d);

        void _gaussian(int n, double a, double* d);

        void _hamming(unsigned n, double* d);

        void _blackman(unsigned n, double* d);

        void _interpolate(const double* x, const double* y,
                unsigned nX, unsigned n, double* result);

        unsigned _nextPowerOf2(unsigned n);

        void _generateFirFilter(unsigned n, double w,
                const double* window, double* result);

};

PELICAN_DECLARE_DATABLOB(PolyphaseCoefficients)

}// namespace ampp
}// namespace pelican

#endif // POLYPHASE_COEFFICIENTS_H_
