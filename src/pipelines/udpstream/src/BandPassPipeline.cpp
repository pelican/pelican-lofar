#include "BandPassPipeline.h"
#include "BandPassRecorder.h"
#include "PPFChanneliser.h"
#include "StokesGenerator.h"


namespace pelican {
namespace lofar {

/**
 *@details BandPassPipeline 
 */
BandPassPipeline::BandPassPipeline( const QString& streamIdentifier )
    : AbstractPipeline(), _streamIdentifier(streamIdentifier)
{
}

/**
 *@details
 */
BandPassPipeline::~BandPassPipeline()
{
}

void BandPassPipeline::init() {
    // Create modules
    _ppfChanneliser = (PPFChanneliser *) createModule("PPFChanneliser");
    _stokesGenerator = (StokesGenerator *) createModule("StokesGenerator");
    _recorder = (BandPassRecorder*) createModule("BandPassRecorder");

    // Create local datablobs
    _spectra = (SpectrumDataSetC32*) createBlob("SpectrumDataSetC32");
    _stokes = (SpectrumDataSetStokes*) createBlob("SpectrumDataSetStokes");

    requestRemoteData(_streamIdentifier);
}

void BandPassPipeline::run(QHash<QString, DataBlob*>& remoteData) {

    _ppfChanneliser->run((TimeSeriesDataSetC32*)remoteData[_streamIdentifier], 
                         _spectra);
    _stokesGenerator->run(_spectra, _stokes);
    if( _recorder->run(_stokes , &_bandPass) ) {
        dataOutput(&_bandPass);
        deactivate(); // job is done
    }
}

} // namespace lofar
} // namespace pelican
