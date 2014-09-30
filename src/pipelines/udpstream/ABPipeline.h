#ifndef ABPIPELINE_H
#define ABPIPELINE_H

#include "pelican/core/AbstractPipeline.h"

namespace pelican {
namespace ampp {

class ABData;
class ABProc;
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
        ABProc* amplifier;

        // Local data blob pointers.
        ABData* outputData;
        unsigned long counter;
};

} // namespace ampp
} // namespace pelican

#endif // ABPIPELINE_H

