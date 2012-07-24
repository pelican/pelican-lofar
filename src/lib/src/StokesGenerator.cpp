#include "StokesGenerator.h"
#include "SpectrumDataSet.h"

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



StokesGenerator::~StokesGenerator()
{
}


/**
 * @details
 *
 */

void StokesGenerator::run(const SpectrumDataSetC32* channeliserOutput,
        SpectrumDataSetStokes* stokes)
{
    typedef std::complex<float> Complex;
    unsigned nSamples = channeliserOutput->nTimeBlocks();
    unsigned nSubbands = channeliserOutput->nSubbands();
    unsigned nChannels = channeliserOutput->nChannels();
    Q_ASSERT( channeliserOutput->nPolarisations() >= 2 );

    stokes->setLofarTimestamp(channeliserOutput->getLofarTimestamp());
    stokes->setBlockRate(channeliserOutput->getBlockRate());
    stokes->resize(nSamples, nSubbands, 4, nChannels);

    const Complex* dataPolX, *dataPolY;
    float *I, *Q, *U, *V;
    float powerX, powerY;
    float XxYstarReal;
    float XxYstarImag;

    const Complex* dataPolDataBlock = channeliserOutput->data();
    for (unsigned t = 0; t < nSamples; ++t) {
        for (unsigned s = 0; s < nSubbands; ++s) {
            unsigned dataPolIndexX = channeliserOutput->index( s, nSubbands, 
                                                               0,2,
                                                               t, nChannels);
            unsigned dataPolIndexY = channeliserOutput->index( s, nSubbands, 
                                                               1,2,
                                                               t, nChannels);
            //unsigned indexStokes = stokes->index( s, nSubbands,
            //                                      0,2,
            //                                      t, nChannels );
            //dataPolX = channeliserOutput->spectrumData(t, s, 0);
            //dataPolY = channeliserOutput->spectrumData(t, s, 1);
            dataPolX = &dataPolDataBlock[dataPolIndexX];
            dataPolY = &dataPolDataBlock[dataPolIndexY];
            // TODO speed up
            I = stokes->spectrumData(t, s, 0);
            Q = stokes->spectrumData(t, s, 1);
            U = stokes->spectrumData(t, s, 2);
            V = stokes->spectrumData(t, s, 3);
            //std::cout << "nSamples=" << nSamples << " t=" << t << " s=" << s;
            //std::cout << "  dataPolIndexX=" << dataPolIndexX;
            //std::cout << "  dataPolIndexY=" << dataPolIndexY << std::endl;
            // std::cout << I << " "<< Q <<" "<< U << " "<< V <<std::endl;
            for (unsigned c = 0; c < nChannels; ++c) {
                float Xr = dataPolX[c].real();
                float Xi = dataPolX[c].imag();
                float Yr = dataPolY[c].real();
                float Yi = dataPolY[c].imag();
                XxYstarReal = Xr*Yr + Xi*Yi;
                XxYstarImag = Xi*Yr - Xr*Yi;

                powerX = _sqr(Xr) + _sqr(Xi);
                powerY = _sqr(Yr) + _sqr(Yi);

                I[c] = powerX + powerY;
                Q[c] = powerX - powerY;
                U[c] = 2.0f * XxYstarReal;
                V[c] = 2.0f * XxYstarImag;
            }
        }
    }
}




/**
 * @details
 * Not used?
 */
void StokesGenerator::run(const TimeSeriesDataSetC32* streamData,
        SpectrumDataSetStokes* stokes)
{
    typedef std::complex<float> Complex;
    unsigned nSamples = streamData->nTimeBlocks();
    unsigned nSubbands = streamData->nSubbands();
    unsigned nSamps = streamData->nTimesPerBlock();

    stokes->setLofarTimestamp(streamData->getLofarTimestamp());
    stokes->setBlockRate(streamData->getBlockRate());
    stokes->resize(nSamples * nSamps, nSubbands, 4, 1);

    const Complex* dataPolX, *dataPolY;
    float XxYstarReal;
    float XxYstarImag;
    float *I, *Q, *U, *V;
    float powerX, powerY;

    for (unsigned t = 0; t < nSamples; ++t) {
        for(unsigned s = 0; s < nSubbands; ++s) {
            dataPolX = streamData->timeSeriesData(t, s, 0);
            dataPolY = streamData->timeSeriesData(t, s, 1);
            for(unsigned c = 0; c < nSamps; ++c) {
                I = stokes->spectrumData(t * nSamps + c, s, 0);
                Q = stokes->spectrumData(t * nSamps + c, s, 1);
                U = stokes->spectrumData(t * nSamps + c, s, 2);
                V = stokes->spectrumData(t * nSamps + c, s, 3);
                // NOTE: We have one channel per subband since we are not channelising
                float Xr = dataPolX[c].real();
                float Xi = dataPolX[c].imag();
                float Yr = dataPolY[c].real();
                float Yi = dataPolY[c].imag();
                XxYstarReal = Xr*Yr + Xi*Yi;
                XxYstarImag = Xi*Yr - Xr*Yi;

                powerX = _sqr(Xr) + _sqr(Xi);
                powerY = _sqr(Yr) + _sqr(Yi);

                I[c] = powerX + powerY;
                Q[c] = powerX - powerY;
                U[c] = 2.0f * XxYstarReal;
                V[c] = 2.0f * XxYstarImag;

            }
        }
    }
}


}// namespace lofar
}// namespace pelican

