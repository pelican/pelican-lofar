#ifndef ABPROC_H
#define ABPROC_H

#include "pelican/core/AbstractModule.h"

namespace pelican {
namespace ampp {

class ABData;

/*
 * A simple example to demonstrate how to write a pipeline module.
 */
class ABProc : public AbstractModule
{
    public:
        // Constructs the module.
        ABProc(const ConfigNode& config);

        // Runs the module.
        void run(const ABData* input, ABData* output); 
    private:
        float gain;
};

PELICAN_DECLARE_MODULE(ABProc)

} // namespace ampp
} // namespace pelican

#endif // ABPROC_H

