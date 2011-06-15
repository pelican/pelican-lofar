#ifndef BANDPASSPIPELINE_H
#define BANDPASSPIPELINE_H


#include "pelican/core/AbstractPipeline.h"
#include "BandPass.h"

/**
 * @file BandPassPipeline.h
 */

namespace pelican {
class DataBlob;
namespace lofar {
class BandPassRecorder;
class PPFChanneliser;
class StokesGenerator;
class SpectrumDataSetC32;
class SpectrumDataSetStokes;

/**
 * @class BandPassPipeline
 *  
 * @brief
 *    pipeline to run a bandpass calibration, producing a suitable BandPass object
 * @details
 * 
 */

class BandPassPipeline : public AbstractPipeline
{
    public:
        BandPassPipeline( const QString& streamIdentifier  );
        ~BandPassPipeline();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
        QString _streamIdentifier;
        BandPassRecorder* _recorder;
        PPFChanneliser* _ppfChanneliser;
        StokesGenerator* _stokesGenerator;
        SpectrumDataSetC32* _spectra;
        SpectrumDataSetStokes* _stokes;
        BandPass _bandPass;
};

} // namespace lofar
} // namespace pelican
#endif // BANDPASSPIPELINE_H 
