#ifndef STOKES_GENERATOR_H
#define STOKES_GENERATOR_H

/**
 * @file StokesGenerator.h
 */

#include "pelican/modules/AbstractModule.h"
#include "TimeSeriesDataSet.h"

#include <vector>
#include <complex>

namespace pelican {

class ConfigNode;

namespace lofar {

class SpectrumDataSetC32;
class SpectrumDataSetStokes;

/**
 * @class StokesGenerator
 *
 * @details Module used for converting a collection of spectra from X,Y polarisations to stokes parameters.

 *
 */

class StokesGenerator : public AbstractModule
{
    public:
        /// Constructor.
        StokesGenerator(const ConfigNode& config);

        /// Destructor.
        ~StokesGenerator();

    public:
        ///
        void run(const SpectrumDataSetC32* channeliserOutput,
                SpectrumDataSetStokes* stokes);

        ///
        void run(const TimeSeriesDataSetC32* streamData,
                SpectrumDataSetStokes* stokes);

    private:
        float _sqr(float x) { return x * x; }
        unsigned _numberOfStokes;
};

// Declare this class as a pelican module.
PELICAN_DECLARE_MODULE(StokesGenerator)

}// namespace lofar
}// namespace pelican

#endif // PPF_CHANNELISER_H_
