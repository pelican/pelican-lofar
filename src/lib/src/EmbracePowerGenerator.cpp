#include "EmbracePowerGenerator.h"
#include "SpectrumDataSet.h"

#include "pelican/utility/ConfigNode.h"

#include <iostream>
#include <cmath>
#include <complex>

namespace pelican {
namespace lofar {


///
EmbracePowerGenerator::EmbracePowerGenerator(const ConfigNode& config)
: AbstractModule(config)
{
}



EmbracePowerGenerator::~EmbracePowerGenerator()
{
}


/**
 * @details
 *
 */

void EmbracePowerGenerator::run(const SpectrumDataSetC32* channeliserOutput,
				SpectrumDataSetStokes* stokes)
{
    typedef std::complex<float> Complex;
    unsigned nSamples = channeliserOutput->nTimeBlocks();
    unsigned nSubbands = channeliserOutput->nSubbands();
    unsigned nChannels = channeliserOutput->nChannels();
    Q_ASSERT( channeliserOutput->nPolarisations() >= 2 );

    stokes->setLofarTimestamp(channeliserOutput->getLofarTimestamp());
    stokes->setBlockRate(channeliserOutput->getBlockRate());
    stokes->resize(nSamples, nSubbands, 2, nChannels); // 2 is because
						       // EMBRACE has
						       // 2, single
						       // pol
						       // directions
						       // rather than
						       // polarizations

    const Complex* dataPolX, *dataPolY;
    float *IX, *IY;
    float powerX, powerY;

    const Complex* dataPolDataBlock = channeliserOutput->data();
    for (unsigned t = 0; t < nSamples; ++t) {
        for (unsigned s = 0; s < nSubbands; ++s) {
            unsigned dataPolIndexX = channeliserOutput->index( s, nSubbands, 
                                                               0,2,
                                                               t, nChannels);
            unsigned dataPolIndexY = channeliserOutput->index( s, nSubbands, 
                                                               1,2,
                                                               t, nChannels);
            dataPolX = &dataPolDataBlock[dataPolIndexX];
            dataPolY = &dataPolDataBlock[dataPolIndexY];

            IX = stokes->spectrumData(t, s, 0);
            IY = stokes->spectrumData(t, s, 1);

            for (unsigned c = 0; c < nChannels; ++c) {
                float Xr = dataPolX[c].real();
                float Xi = dataPolX[c].imag();
                float Yr = dataPolY[c].real();
                float Yi = dataPolY[c].imag();

                powerX = _sqr(Xr) + _sqr(Xi);
                powerY = _sqr(Yr) + _sqr(Yi);

                IX[c] = powerX;
                IY[c] = powerY;
            }
        }
    }
}

}// namespace lofar
}// namespace pelican

