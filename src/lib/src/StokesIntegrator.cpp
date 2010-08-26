#include "StokesIntegrator.h"
#include "SpectrumDataSet.h"

#include "pelican/utility/pelicanTimer.h"
#include "pelican/utility/ConfigNode.h"

#include <iostream>
#include <cmath>

namespace pelican {
namespace lofar {


///
StokesIntegrator::StokesIntegrator(const ConfigNode& config)
: AbstractModule(config)
{
    // Get the size for the integration window(step) from the parameter file.

    _windowSize    = config.getOption("integrateTimeBins", "value", "1").toUInt();
}


///
StokesIntegrator::~StokesIntegrator()
{
}


///
void StokesIntegrator::run(const SpectrumDataSetStokes* stokesGeneratorOutput,
        SpectrumDataSetStokes* intStokes)
{
    unsigned nSamples = stokesGeneratorOutput->nTimeBlocks();
    unsigned nSubbands = stokesGeneratorOutput->nSubbands();
    unsigned nChannels = stokesGeneratorOutput->nChannels(0);
    unsigned nPols = stokesGeneratorOutput->nPolarisations();
    //std::cout << "nSamples= " << nSamples << std::endl;
    //std::cout << "nSubbands= " << nSubbands << std::endl;
    //std::cout << "nChannels= " << nChannels << std::endl;

    //TIMER_ENABLE

    //    intStokes -> setLofarTimestamp(stokesGeneratorOutput -> getLofarTimestamp());
    //    intStokes -> setBlockRate(stokesGeneratorOutput -> getBlockRate());

    // Checking if the integration window is bigger than the available samples. In case
    // // it is greater a warning is produced and the window size is reduced to the size
    // // of the samples.

    if (_windowSize>nSamples){
        std::cout << "Warning the window size has been reduced from" << _windowSize << "to the total sample count" << nSamples << std::endl;
        _windowSize=nSamples;
    }
    //std::cout << "_windowSize= " << _windowSize << std::endl;

    unsigned newSamples = nSamples/_windowSize;
    const float* value;
    float* value2;
    //    unsigned timeFloats = nPols*nSubbands*nChannels;

    intStokes->resize(newSamples, nSubbands, nPols, nChannels);
    for (unsigned i = 0; i < newSamples * nSubbands * nPols; ++i) {
        value2 = intStokes->spectrum(i)->ptr();
        for (unsigned c = 0; c < nChannels; ++c) value2[c] = 0.0;
    }

    unsigned timeStart=0;
    unsigned bufferCounter;
    //unsigned ts;
    //TIMER_START;

    for (unsigned u = 0; u < newSamples; ++u) {
        for (unsigned t = timeStart; t < _windowSize+timeStart; ++t) {
            for (unsigned p = 0; p < nPols; ++p) {
                for (unsigned s = 0; s < nSubbands; ++s) {
                    value = stokesGeneratorOutput->spectrumData(t, s, p);
                    float* timeBuffer = intStokes->spectrumData(u,s,p);
                    for (unsigned c = 0; c < nChannels; ++c){
                        timeBuffer[c]+= value[c];
                        bufferCounter++;
                    }
                }
            }
        }
        timeStart=timeStart+_windowSize;
    }

    //TIMER_STOP(ts);
    //std::cout << ts << std::endl;
}

}// namespace lofar
}// namespace pelican

