#ifndef SIGPROCPIPELINE_H
#define SIGPROCPIPELINE_H


#include "pelican/core/AbstractPipeline.h"
#include "RFI_Clipper.h"
#include "DedispersionModule.h"
#include "DedispersionAnalyser.h"
#include "SigprocStokesWriter.h"
#include "DedispersionDataAnalysisOutput.h"
#include "WeightedSpectrumDataSet.h"

/**
 * @file SigprocPipeline.h
 */

namespace pelican {

namespace ampp {

/**
 * @class SigprocPipeline
 *
 * @brief
 *    Reads in sigproc files in to a pipeline
 * @details
 *
 */

class SigprocPipeline : public AbstractPipeline
{
    public:
        SigprocPipeline();
        ~SigprocPipeline();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

        /// called internally to free up DataBlobs after they are finished with
        void updateBufferLock( const QList<DataBlob*>& );

    protected:
        void dedispersionAnalysis( DataBlob* data );

    private:
        RFI_Clipper* _rfiClipper;
        DedispersionModule* _dedispersionModule;
        DedispersionAnalyser* _dedispersionAnalyser;

        QList<SpectrumDataSetStokes*> _stokesData;
        LockingPtrContainer<SpectrumDataSetStokes>* _stokesBuffer;
        WeightedSpectrumDataSet* _weightedIntStokes;

        unsigned int _minEventsFound;
        unsigned int _maxEventsFound;
};

} // namespace ampp
} // namespace pelican
#endif // SIGPROCPIPELINE_H
