#include "StokesGenerator.h"
#include "SubbandSpectra.h"
#include "Spectrum.h"

#include "pelican/utility/ConfigNode.h"

#include <iostream>
#include <cmath>
#include <complex>

namespace pelican {
namespace lofar {


///
StokesGenerator::StokesGenerator(const ConfigNode& config)
: AbstractModule(config)
{
}


///
StokesGenerator::~StokesGenerator()
{
}

///
void StokesGenerator::run(const SubbandSpectraC32* channeliserOutput,
        SubbandSpectraStokes* stokes)
{
    typedef std::complex<float> Complex;
    unsigned nSamples = channeliserOutput->nTimeBlocks();
    unsigned nSubbands = channeliserOutput->nSubbands();
    unsigned nChannels = channeliserOutput->ptr(0,0,0)->nChannels();
    float powerX = 0.0, powerY = 0.0;

    stokes->resize(nSamples, nSubbands, 4, nChannels);

    for (unsigned t = 0; t < nSamples; ++t) {
        for (unsigned s = 0; s < nSubbands; ++s) {
            const Spectrum<Complex>* spectPolX = channeliserOutput->ptr(t, s, 0);
            const Spectrum<Complex>* spectPolY = channeliserOutput->ptr(t, s, 1);
            const Complex* dataPolX = spectPolX->ptr();
            const Complex* dataPolY = spectPolY->ptr();
            float* I = stokes->ptr(t, s, 0)->ptr();
            float* Q = stokes->ptr(t, s, 1)->ptr();
            float* U = stokes->ptr(t, s, 2)->ptr();
            float* V = stokes->ptr(t, s, 3)->ptr();
            for (unsigned c = 0; c < nChannels; ++c) {
                powerX = _sqr(dataPolX[c].real()) + _sqr(dataPolX[c].imag());
                powerY = _sqr(dataPolY[c].real()) + _sqr(dataPolY[c].imag());
                I[c] = powerX + powerY;
                Q[c] = powerX - powerY;
                U[c] = 2.0 * real(dataPolX[c] * conj(dataPolY[c]));
                V[c] = 2.0 * imag(dataPolX[c] * conj(dataPolY[c]));
//                I[c] = sin(2 * M_PI * 2.0 * c * 0.1);//powerX + powerY;
//                Q[c] = c;//powerX - powerY;
//                U[c] = 2*c;//2.0 * real(dataPolX[c] * conj(dataPolY[c]));
//                V[c] = 3+3*c;//2.0 * imag(dataPolX[c] * conj(dataPolY[c]));
            }
        }
    }
}


}// namespace lofar
}// namespace pelican

