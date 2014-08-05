#include "StokesIntegrator.h"
#include "SpectrumDataSet.h"

#include "pelican/utility/pelicanTimer.h"
#include "pelican/utility/ConfigNode.h"

#include <iostream>
#include <cmath>

namespace pelican {
namespace ampp {


///
StokesIntegrator::StokesIntegrator(const ConfigNode& config)
: AbstractModule(config)
{
    // Get the size for the integration window(step) from the parameter file.

    _windowSize    = config.getOption("integrateTimeBins", "value", "1").toUInt();
    _binChannels    = config.getOption("integrateFrequencyChannels", "value", "1").toUInt();
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
    unsigned nChannels = stokesGeneratorOutput->nChannels();
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
    unsigned newChannels = nChannels / _binChannels;
    const float* value;
    //    unsigned timeFloats = nPols*nSubbands*nChannels;

    //    intStokes->resize(newSamples, nSubbands, nPols, nChannels);
    intStokes->resize(newSamples, nSubbands, nPols, newChannels);

    // Not sure what business this has here
    float* value2;
    value2 = intStokes->data();
    for (unsigned i = 0; i < newSamples * nSubbands * nPols * newChannels; ++i)
        value2[i] = 0.0;


    //    unsigned bufferCounter;
    //unsigned ts;
    //TIMER_START;
    /* Code before frequency integrator
    for (unsigned u = 0; u < newSamples; ++u) {
      for (unsigned t = timeStart; t < _windowSize+timeStart; ++t) {
	for (unsigned s = 0; s < nSubbands; ++s) {
	  for (unsigned p = 0; p < nPols; ++p) {
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
    */
    unsigned timeStart=0;
    unsigned channelStart=0;

    for (unsigned u = 0; u < newSamples; ++u) {
      for (unsigned t = timeStart; t < _windowSize+timeStart; ++t) {
	for (unsigned s = 0; s < nSubbands; ++s) {
	  for (unsigned p = 0; p < nPols; ++p) {
	    value = stokesGeneratorOutput->spectrumData(t, s, p);
	    float* timeBuffer = intStokes->spectrumData(u,s,p);
	    channelStart = 0;
	    for (unsigned nc = 0; nc < newChannels; ++nc){
	      for (unsigned c = channelStart; c < channelStart + _binChannels; ++c){
		timeBuffer[nc]+= value[c];
	      }
	      channelStart += _binChannels;
	    }
	  }
	}
      }
      timeStart=timeStart+_windowSize;
    }

    // Set the timestamp of the first time sample
    intStokes->setLofarTimestamp(stokesGeneratorOutput->getLofarTimestamp());

    //std::cout << "timestamp in integrator:" << intStokes->getLofarTimestamp() << std::endl;
    //TIMER_STOP(ts);
    //std::cout << ts << std::endl;
}

}// namespace ampp
}// namespace pelican

