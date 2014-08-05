#ifndef EMBRACE_BF_PIPELINE_H
#define EMBRACE_BF_PIPELINE_H


#include "pelican/core/AbstractPipeline.h"
#include "pelican/data/DataBlob.h"
#include "pelican/output/PelicanTCPBlobServer.h"

#include "PPFChanneliser.h"
#include "EmbracePowerGenerator.h"
#include "RFI_Clipper.h"
#include "StokesIntegrator.h"

#include "AdapterTimeSeriesDataSet.h"
#include "TimeSeriesDataSet.h"
#include "SpectrumDataSet.h"

#include "EmbraceFBWriter.h"

/**
 * @file EmbraceBFPipeline.h
 */

namespace pelican {
namespace ampp {

/**
 * @class EmbraceBFPipeline
 *
 * @brief The EmbraceBF Pipeline.
 *
 * @details
 *
 */
class EmbraceBFPipeline : public AbstractPipeline
{
    public:
	/// Constructor
        EmbraceBFPipeline( const QString& streamIdentifier );
	/// Destructor
        ~EmbraceBFPipeline();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
	int _totalIterations;
        QString _streamIdentifier;

        /// Module pointers
        PPFChanneliser* ppfChanneliser;
        EmbracePowerGenerator* embracePowerGenerator;
        StokesIntegrator* stokesIntegrator;
        RFI_Clipper* rfiClipper;

        /// Local data blob
        SpectrumDataSetC32* spectra;
        TimeSeriesDataSetC32* timeSeries;
        SpectrumDataSetStokes* stokes;
        SpectrumDataSetStokes* intStokes;
        WeightedSpectrumDataSet* weightedIntStokes;

        unsigned _iteration;
};

} // namespace ampp
} // namespace pelican

#endif // EMBRACE_BF_PIPELINE_H
