#ifndef STOKES_GENERATOR_H
#define STOKES_GENERATOR_H

/**
 * @file StokesGenerator.h
 */

#include "pelican/modules/AbstractModule.h"
#include "SubbandTimeSeries.h"

#include <vector>
#include <complex>

namespace pelican {

class ConfigNode;

namespace lofar {

class SubbandSpectraC32;
class SubbandSpectraStokes;

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
        void run(const SubbandSpectraC32* channeliserOutput,
                SubbandSpectraStokes* stokes);
        void run(const SubbandTimeSeriesC32* streamData,
                SubbandSpectraStokes* stokes);

    private:
        float _sqr(float x) { return x * x; }

};

// Declare this class as a pelican module.
PELICAN_DECLARE_MODULE(StokesGenerator)

}// namespace lofar
}// namespace pelican

#endif // PPF_CHANNELISER_H_
