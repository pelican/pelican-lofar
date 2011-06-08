#ifndef UDP_BF_PIPELINE_H
#define UDP_BF_PIPELINE_H

#include "RFI_Clipper.h"
#include "pelican/core/AbstractPipeline.h"
#include "pelican/data/DataBlob.h"
#include "pelican/output/PelicanTCPBlobServer.h"

#include "PPFChanneliser.h"
#include "StokesGenerator.h"
#include "StokesIntegrator.h"

#include "AdapterTimeSeriesDataSet.h"
#include "TimeSeriesDataSet.h"
#include "SpectrumDataSet.h"

#include "SigprocStokesWriter.h"
#include "timer.h"

/**
 * @file UdpBFPipeline.h
 */

namespace pelican {
namespace lofar {

/**
 * @class UdpBFPipeline
 *
 * @brief
 *
 * @details
 *
 */
class TimingPipeline : public AbstractPipeline
{
    public:
        TimingPipeline();
        ~TimingPipeline();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
        /// Module pointers
        PPFChanneliser* ppfChanneliser;
        StokesGenerator* stokesGenerator;
        StokesIntegrator* stokesIntegrator;
        RFI_Clipper* rfiClipper;

        /// Local data blob
        SpectrumDataSetC32* spectra;
        TimeSeriesDataSetC32* timeSeries;
        SpectrumDataSetStokes* stokes;
        SpectrumDataSetStokes* intStokes;

        unsigned _totalSamplesPerChunk;
        unsigned _iteration;

        // Timers.
        TimerData _ppfTime;
        TimerData _stokesTime;
        TimerData _integratorTime;
        TimerData _outputTime;
	TimerData _totalTime;
	TimerData _rfiClipper;
};

} // namespace lofar
} // namespace pelican

#endif // UDP_BF_PIPELINE_H
