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
        qint64 _chunkSize;
        qint64 _bytesRead;
};

PELICAN_DECLARE_CHUNKER(ABChunker)

} // namespace ampp
} // namespace pelican

#endif // ABCHUNKER_H

