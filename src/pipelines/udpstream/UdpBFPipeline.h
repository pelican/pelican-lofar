#ifndef UDP_BF_PIPELINE_H
#define UDP_BF_PIPELINE_H


#include "pelican/core/AbstractPipeline.h"
#include "pelican/data/DataBlob.h"
#include "pelican/output/PelicanTCPBlobServer.h"

#include "PPFChanneliser.h"
#include "StokesGenerator.h"
#include "RFI_Clipper.h"
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
namespace ampp {

/**
 * @class UdpBFPipeline
 *
 * @brief The UdpBF Pipeline.
 *
 * @details
 *
 */
class UdpBFPipeline : public AbstractPipeline
{
    public:
	/// Constructor
        UdpBFPipeline( const QString& streamIdentifier );
	/// Destructor
        ~UdpBFPipeline();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
	int _totalIterations;
        QString _streamIdentifier;

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
        WeightedSpectrumDataSet* weightedIntStokes;

        unsigned _iteration;
#ifdef TIMING_ENABLED
        TimerData _totalTime;
        TimerData _rfiClipperTime;
#endif


};

} // namespace ampp
} // namespace pelican

#endif // UDP_BF_PIPELINE_H
