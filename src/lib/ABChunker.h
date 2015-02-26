#ifndef ABCHUNKER_H
#define ABCHUNKER_H

#include "pelican/server/AbstractChunker.h"

namespace pelican {
namespace ampp {

/*
 * A simple example to demonstrate how to write a data chunker.
 */
class ABChunker : public AbstractChunker
{
    public:
        // Constructs the chunker.
        ABChunker(const ConfigNode& config);

        // Creates the input device (usually a socket).
        virtual QIODevice* newDevice();

        // Obtains a chunk of data from the device when data is available.
        virtual void next(QIODevice*);
    private:
        unsigned long int _chunksProced;
        unsigned int _chunkSize;
        unsigned int _pktSize;
        unsigned int _hdrSize;
        unsigned int _ftrSize;
        unsigned int _payloadSize;
        unsigned int _pktsPerSpec;
        unsigned int _nPackets;
        unsigned int _first;
        unsigned int _numMissInst;
        unsigned int _numMissPkts;
        unsigned int _x;
        unsigned int _y;
};

PELICAN_DECLARE_CHUNKER(ABChunker)

} // namespace ampp
} // namespace pelican

#endif // ABCHUNKER_H

