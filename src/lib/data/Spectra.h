#ifndef SPECTRA_H_
#define SPECTRA_H_

#include "data/Matrix.h"
#include "data/Spectrum.h"

#include <complex>

using std::complex;

namespace pelican {
namespace lofar {

/**
 * @class Spectra
 *
 * @brief
 *
 * @details
 *
 */

template <typename T>
class Spectra : public DataBlob
{
    private:
        Matrix<Spectrum<T> > _spectra;

    public:
        Spectra() : DataBlob("Spectra") {}

        Spectra(unsigned nSubbands, unsigned nPolarisations, unsigned nChannels,
                unsigned nBlocks = 1)
        : DataBlob("Spectra")
        {
            _spectra.resize(nSubbands, nPolarisations);
        }

        ~Spectra() {}

    public:
        unsigned nSubbands() { return _spectra.nRows(); }

        unsigned nPolarisations() { return _spectra.nColumns(); }

        void setSpectrumSize(unsigned s, unsigned p, unsigned nChannels) {
            _spectra[s][p].resize(nChannels * nBlocks);
        }

        void setSpectrum(unsigned s, unsigned p, unsigned nChannels) {
            _spectra[s][p].resize(nChannels);
        }

        Spectrum<T>& getSpectrum(unsigned s, unsigned p)
        {
            return _spectra[s][p];
        }
};


} // namespace lofar
} // namespace pelican
#endif // LOFARDATA_H_
