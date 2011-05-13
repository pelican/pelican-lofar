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

namespace lofar {

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
        RFI_Clipper* rfiClipper;
};

} // namespace lofar
} // namespace pelican
#endif // STOKESONLYPIPELINE_H 
