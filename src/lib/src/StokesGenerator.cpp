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
 * Converts a collection of spectra from x,y polarisation to stokes
 * parameters.
 */
void StokesGenerator::run(const SpectrumDataSetC32* channeliserOutput,
        SpectrumDataSetStokes* stokes)
{
    typedef std::complex<float> Complex;
    unsigned nSamples = channeliserOutput->nTimeBlocks();
    unsigned nSubbands = channeliserOutput->nSubbands();
    unsigned nChannels = channeliserOutput->nChannels();

    stokes->setLofarTimestamp(channeliserOutput->getLofarTimestamp());
    stokes->setBlockRate(channeliserOutput->getBlockRate());
    stokes->resize(nSamples, nSubbands, 1, nChannels);

    const Complex* dataPolX, *dataPolY;
    float *I, *Q, *U, *V;
    float powerX, powerY;
    Complex XxYstar;

    for (unsigned t = 0; t < nSamples; ++t) {
        for (unsigned s = 0; s < nSubbands; ++s) {
            dataPolX = channeliserOutput->spectrumData(t, s, 0);
            dataPolY = channeliserOutput->spectrumData(t, s, 1);
            I = stokes->spectrumData(t, s, 0);
	    //	    Q = stokes->spectrumData(t, s, 1);
	    //	    U = stokes->spectrumData(t, s, 2);
	    //	    V = stokes->spectrumData(t, s, 3);
            // std::cout << nSamples << " " << t << " " << s  <<std::endl;
            // std::cout << dataPolX << " " << dataPolY << std::endl;
            // std::cout << I << " "<< Q <<" "<< U << " "<< V <<std::endl;
            for (unsigned c = 0; c < nChannels; ++c) {
                // XxYstar=dataPolX[c]*conj(dataPolY[c]);
                powerX = _sqr(dataPolX[c].real()) + _sqr(dataPolX[c].imag());
                powerY = _sqr(dataPolY[c].real()) + _sqr(dataPolY[c].imag());
                I[c] = powerX + powerY;
		//		Q[c] = powerX - powerY;
		//		U[c] = 2.0f * real(XxYstar);
		//		V[c] = 2.0f * imag(XxYstar);
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
                powerX = _sqr(dataPolX[0].real()) + _sqr(dataPolX[0].imag());
                powerY = _sqr(dataPolY[0].real()) + _sqr(dataPolY[0].imag());
                I[0] = powerX + powerY;
                Q[0] = powerX - powerY;
                U[0] = 2.0f * real(dataPolX[0] * conj(dataPolY[0]));
                V[0] = 2.0f * imag(dataPolX[0] * conj(dataPolY[0]));
            }
        }
    }
}


}// namespace lofar
}// namespace pelican

