#ifndef STOKES_INTEGRATOR_H
#define STOKES_INTEGRATOR_H

/**
 * @file StokesIntegrator.h
 */

#include "pelican/modules/AbstractModule.h"
#include "TimeSeriesDataSet.h"
#include "pelican/utility/ConfigNode.h"

#include <vector>
#include <complex>

namespace pelican {

class ConfigNode;

namespace lofar {

//class SubbandSpectraC32;
class SpectrumDataSetStokes;

/**
 * @class StokesIntegrator
 *
 * @details Retrieves desired integration step size from the config file and applies it to the stokes data.
 * 
 * Relevant config option from xml file 
 @verbatim
 code
 @endverbatim
 *
 */

class StokesIntegrator : public AbstractModule
{
    public:
        /// Constructor
        StokesIntegrator(const ConfigNode& config);

        /// Destructor
        ~StokesIntegrator();

        ///
        void run(const SpectrumDataSetStokes* stokesGeneratorOutput, SpectrumDataSetStokes* intStokes);
        //	void run(const SubbandTimeSeriesC32* streamData,
        //      	SubbandSpectraStokes* stokes);

    private:
        unsigned _windowSize;
        unsigned timeStart;
};


// Declare this class as a pelican module.
PELICAN_DECLARE_MODULE(StokesIntegrator)

}// namespace lofar
}// namespace pelican
#endif // STOKES_INTEGRATOR_H
