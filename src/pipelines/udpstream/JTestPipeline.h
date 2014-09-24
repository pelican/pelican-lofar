#ifndef JTESTPIPELINE_H
#define JTESTPIPELINE_H

#include "pelican/core/AbstractPipeline.h"

namespace pelican {
namespace ampp {

class JTestData;
class JTestProc;
class JTestPipeline : public AbstractPipeline
{
    public:
        // Constructor.
        JTestPipeline();

        // Destructor
        ~JTestPipeline();

        // Initialises the pipeline.
        void init();

        // Defines one iteration of the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
        // Module pointers.
        JTestProc* amplifier;

        // Local data blob pointers.
        JTestData* outputData;
        unsigned long counter;
};

} // namespace ampp
} // namespace pelican

#endif // JTESTPIPELINE_H

