#ifndef JTESTCHUNKER_H
#define JTESTCHUNKER_H

#include "pelican/server/AbstractChunker.h"

namespace pelican {
namespace ampp {

/*
 * A simple example to demonstrate how to write a data chunker.
 */
class JTestChunker : public AbstractChunker
{
    public:
        // Constructs the chunker.
        JTestChunker(const ConfigNode& config);

        // Creates the input device (usually a socket).
        virtual QIODevice* newDevice();

        // Obtains a chunk of data from the device when data is available.
        virtual void next(QIODevice*);
    private:
        qint64 _chunkSize;
        qint64 _bytesRead;
};

PELICAN_DECLARE_CHUNKER(JTestChunker)

} // namespace ampp
} // namespace pelican

#endif // JTESTCHUNKER_H

