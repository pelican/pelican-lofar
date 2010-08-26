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
 * @brief
 *
 * @details
 *
 */

class StokesGenerator : public AbstractModule
{
    public:
        ///
        StokesGenerator(const ConfigNode& config);

        ///
        ~StokesGenerator();

        ///
        void run(const SpectrumDataSetC32* channeliserOutput,
                SpectrumDataSetStokes* stokes);
        void run(const TimeSeriesDataSetC32* streamData,
                SpectrumDataSetStokes* stokes);

    private:
        float _sqr(float x) { return x * x; }

};

// Declare this class as a pelican module.
PELICAN_DECLARE_MODULE(StokesGenerator)

}// namespace lofar
}// namespace pelican

#endif // PPF_CHANNELISER_H_
