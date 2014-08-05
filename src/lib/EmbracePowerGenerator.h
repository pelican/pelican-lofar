#ifndef EMBRACE_POWER_GENERATOR_H
#define EMBRACE_POWER_GENERATOR_H

/**
 * @file EmbracePowerGenerator.h
 */

#include "pelican/modules/AbstractModule.h"
#include "TimeSeriesDataSet.h"

#include <vector>
#include <complex>

namespace pelican {

class ConfigNode;

namespace ampp {

class SpectrumDataSetC32;
class SpectrumDataSetStokes;

/**
 * @class EmbracePowerGenerator
 *
 * @details Module used for converting a collection of spectra from X,Y polarisations to stokes parameters.

 *
 */

class EmbracePowerGenerator : public AbstractModule
{
    public:
        /// Constructor.
        EmbracePowerGenerator(const ConfigNode& config);

        /// Destructor.
        ~EmbracePowerGenerator();

    public:
        ///
        void run(const SpectrumDataSetC32* channeliserOutput,
                SpectrumDataSetStokes* stokes);

    private:
        float _sqr(float x) { return x * x; }

};

// Declare this class as a pelican module.
PELICAN_DECLARE_MODULE(EmbracePowerGenerator)

}// namespace ampp
}// namespace pelican

#endif // PPF_CHANNELISER_H_
