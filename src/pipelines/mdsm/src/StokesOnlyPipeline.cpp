#include "StokesOnlyPipeline.h"


namespace pelican {

namespace lofar {


/**
 *@details StokesOnlyPipeline 
 */
StokesOnlyPipeline::StokesOnlyPipeline()
    : AbstractPipeline()
{
}

/**
 *@details
 */
StokesOnlyPipeline::~StokesOnlyPipeline()
{
}

void StokesOnlyPipeline::init()
{
    // Create modules
    rfiClipper = (RFI_Clipper *) createModule("RFI_Clipper");
    stokesIntegrator = (StokesIntegrator *) createModule("StokesIntegrator");

    // Create local datablobs
    _intStokes = (SpectrumDataSetStokes*) createBlob("SpectrumDataSetStokes");

    // Request remote data
    requestRemoteData("LofarTimeStream1");
}

void StokesOnlyPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    SpectrumDataSetStokes* stokes = (SpectrumDataSetStokes*) remoteData["SpectrumStokes"];

    // Clips RFI and modifies blob in place
    rfiClipper->run(stokes);

    stokesIntegrator->run(stokes, _intStokes);

    // Calls output stream managed->send(data, stream) the output stream
    // manager is configured in the xml.
    dataOutput(_intStokes, "SpectrumDataSetStokes");

}

} // namespace lofar
} // namespace pelican
