#ifndef STOKESONLYPIPELINE_H
#define STOKESONLYPIPELINE_H


#include "pelican/core/AbstractPipeline.h"
#include "pelican/data/DataBlob.h"
#include "RFI_Clipper.h"
#include "StokesIntegrator.h"
#include "SpectrumDataSet.h"
#include "SigprocStokesWriter.h"
#include "FilterBankAdapter.h"

/**
 * @file StokesOnlyPipeline.h
 */

namespace pelican {

namespace ampp {
class WeightedSpectrumDataSet;

/**
 * @class StokesOnlyPipeline
 *  
 * @brief
 *    Pipeline that requires pre-generated stokes 
 *    data, for testing purposes
 * @details
 * 
 */

class StokesOnlyPipeline : public AbstractPipeline
{
    public:
        StokesOnlyPipeline(  );
        ~StokesOnlyPipeline();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
        SpectrumDataSetStokes* _intStokes;
        StokesIntegrator* stokesIntegrator;
        WeightedSpectrumDataSet* _weightedIntStokes;
        RFI_Clipper* rfiClipper;
};

} // namespace ampp
} // namespace pelican
#endif // STOKESONLYPIPELINE_H 
