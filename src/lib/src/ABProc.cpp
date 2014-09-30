#include "ABProc.h"
#include "ABData.h"
#include "pelican/utility/Config.h"

using namespace pelican;
using namespace ampp;

// Construct the example module.
ABProc::ABProc(const ConfigNode& config)
    : AbstractModule(config)
{
    // Set amplifier gain from the XML configuration.
    gain = config.getOption("gain", "value").toDouble();
}

// Runs the module.
void ABProc::run(const ABData* input, ABData* output)
{
    // Ensure the output storage data is big enough.
    unsigned nPts = input->size();
    if (output->size() != nPts)
        output->resize(nPts);

    // Get pointers to the memory to use from the data blobs.
    const float* in = input->ptr();
    float* out = output->ptr();

    // Perform the operation.
    for (unsigned i = 0; i < nPts; ++i) {
        out[i] = gain * in[i];
    }
}

