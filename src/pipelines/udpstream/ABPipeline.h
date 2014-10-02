#ifndef ABPIPELINE_H
#define ABPIPELINE_H

#include "pelican/core/AbstractPipeline.h"

namespace pelican {
namespace ampp {

class ABPipeline : public AbstractPipeline
{
    public:
        // Constructor.
        ABPipeline();

        // Destructor
        ~ABPipeline();

        // Initialises the pipeline.
        void init();

        // Defines one iteration of the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
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

