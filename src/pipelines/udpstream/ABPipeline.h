#ifndef ABPIPELINE_H
#define ABPIPELINE_H

#include "pelican/core/AbstractPipeline.h"
#include "RFI_Clipper.h"
#include "DedispersionModule.h"
#include "DedispersionAnalyser.h"
#include "WeightedSpectrumDataSet.h"

namespace pelican {
namespace ampp {

//class DedispersionModule;

class ABPipeline : public AbstractPipeline
{
    public:
        // Constructor.
        ABPipeline(const QString& streamIdentifier);

        // Destructor
        ~ABPipeline();

        // Initialises the pipeline.
        void init();

        // Defines one iteration of the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

        /// called internally to free up DataBlobs after they are finished with
        void updateBufferLock( const QList<DataBlob*>& );

    protected:
        void dedispersionAnalysis( DataBlob* data );

    private:
        QString _streamIdentifier;

        // Module pointers.
        RFI_Clipper* _rfiClipper;
        DedispersionModule* _dedispersionModule;
        DedispersionAnalyser* _dedispersionAnalyser;

        // Local data blob pointers.
        WeightedSpectrumDataSet* _weightedIntStokes;

        unsigned long counter;
        unsigned _iteration;
	unsigned int _minEventsFound;
	unsigned int _maxEventsFound;
};

} // namespace ampp
} // namespace pelican

#endif // ABPIPELINE_H

