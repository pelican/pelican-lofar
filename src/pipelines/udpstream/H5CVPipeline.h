#ifndef H5_CV_PIPELINE_H
#define H5_CV_PIPELINE_H


#include "pelican/core/AbstractPipeline.h"
#include "pelican/data/DataBlob.h"
#include "pelican/output/PelicanTCPBlobServer.h"

#include "PPFChanneliser.h"
//#include "StokesGenerator.h"
//#include "RFI_Clipper.h"
//#include "StokesIntegrator.h"

#include "AdapterTimeSeriesDataSet.h"
#include "TimeSeriesDataSet.h"
#include "SpectrumDataSet.h"
#include "H5_LofarBFVoltageWriter.h"
#include "SigprocStokesWriter.h"

/**
 * @file H5CVPipeline.h
 */

namespace pelican {
namespace lofar {

/**
 * @class H5CVPipeline
 *
 * @brief The H5CV Pipeline.
 *
 * @details
 *
 */
class H5CVPipeline : public AbstractPipeline
{
    public:
	/// Constructor
        H5CVPipeline( const QString& streamIdentifier );
	/// Destructor
        ~H5CVPipeline();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
        QString _streamIdentifier;

        /// Module pointers
        PPFChanneliser* ppfChanneliser;
        //        StokesGenerator* stokesGenerator;
        //        StokesIntegrator* stokesIntegrator;
        //        RFI_Clipper* rfiClipper;

        /// Local data blob
        SpectrumDataSetC32* spectra;
        TimeSeriesDataSetC32* timeSeries;
        //        SpectrumDataSetStokes* stokes;
        //        SpectrumDataSetStokes* intStokes;
        //        WeightedSpectrumDataSet* weightedIntStokes;

        unsigned _iteration;
};

} // namespace lofar
} // namespace pelican

#endif // H5_CV_PIPELINE_H
