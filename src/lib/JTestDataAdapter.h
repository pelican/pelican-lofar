#ifndef JTESTDATAADAPTER_H
#define JTESTDATAADAPTER_H

#include "pelican/core/AbstractStreamAdapter.h"

namespace pelican {
namespace ampp {

/*
 * Adapter to convert chunks of signal stream data into a JTestData data-blob.
 */
class JTestDataAdapter : public AbstractStreamAdapter
{
    public:
        // Constructs the adapter.
        JTestDataAdapter(const ConfigNode& config);

        // Method to deserialise chunks of memory provided by the I/O device.
        void deserialise(QIODevice* device);

    private:
        static const unsigned _headerSize = 32;
        unsigned _samplesPerPacket;
        unsigned _packetSize;
};

PELICAN_DECLARE_ADAPTER(JTestDataAdapter)

} // namespace ampp
} // namespace pelican

#endif // JTESTDATAADAPTER_H

