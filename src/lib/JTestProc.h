#ifndef JTESTPROC_H
#define JTESTPROC_H

#include "pelican/core/AbstractModule.h"

namespace pelican {
namespace ampp {

class JTestData;

/*
 * A simple example to demonstrate how to write a pipeline module.
 */
class JTestProc : public AbstractModule
{
    public:
        // Constructs the module.
        JTestProc(const ConfigNode& config);

        // Runs the module.
        void run(const JTestData* input, JTestData* output); 
    private:
        float gain;
};

PELICAN_DECLARE_MODULE(JTestProc)

} // namespace ampp
} // namespace pelican

#endif // JTESTPROC_H

