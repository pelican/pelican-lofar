#ifndef ABDATAADAPTER_H
#define ABDATAADAPTER_H

#include "pelican/core/AbstractStreamAdapter.h"

namespace pelican {
namespace ampp {

/*
 * Adapter to convert chunks of signal stream data into a ABData data-blob.
 */
class ABDataAdapter : public AbstractStreamAdapter
{
    public:
        // Constructs the adapter.
        ABDataAdapter(const ConfigNode& config);

        // Method to deserialise chunks of memory provided by the I/O device.
        void deserialise(QIODevice* device);

    private:
        static const unsigned _headerSize = 8;
        static const unsigned _footerSize = 8;
        unsigned _packetsPerSpectrum;
        unsigned _channelsPerPacket;
        unsigned _packetSize;
        unsigned int _nPolarisations;
        unsigned int _nSubbands;
        unsigned int _nChannels;
};

PELICAN_DECLARE_ADAPTER(ABDataAdapter)

} // namespace ampp
} // namespace pelican

#endif // ABDATAADAPTER_H

