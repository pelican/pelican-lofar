#ifndef SIGPROCPIPELINE_H
#define SIGPROCPIPELINE_H


#include "pelican/core/AbstractPipeline.h"
#include "RFI_Clipper.h"
#include "StokesIntegrator.h"
#include "SigprocStokesWriter.h"
#include "WeightedSpectrumDataSet.h"

/**
 * @file SigProcPipeline.h
 */

namespace pelican {

namespace ampp {

/**
 * @class SigProcPipeline
 *  
 * @brief
 *    Reads in sigproc files in to a pipeline
 * @details
 * 
 */

class SigProcPipeline : public AbstractPipeline
{
    public:
        SigProcPipeline(  );
        ~SigProcPipeline();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
        RFI_Clipper* rfiClipper;
        StokesIntegrator* stokesIntegrator;

        SpectrumDataSetStokes* intStokes;
        WeightedSpectrumDataSet* weightedIntStokes;
};

} // namespace ampp
} // namespace pelican
#endif // SIGPROCPIPELINE_H 
