#ifndef EMBRACECHUNKER_H
#define EMBRACECHUNKER_H


#include "pelican/server/AbstractChunker.h"

/**
 * @file EmbraceChunker.h
 */

namespace pelican {

namespace ampp {

/**
 * @class EmbraceChunker
 *  
 * @brief
 *    A chunker for the Embrace telescope RSP Board
 * @details
 * 
 */

class EmbraceChunker : public AbstractChunker
{
    public:
        EmbraceChunker( const ConfigNode& config  );
        ~EmbraceChunker();

        /// Creates the socket to use for the incoming data stream.
        virtual QIODevice* newDevice();

        /// Called whenever there is data ready to be processed.
        virtual void next(QIODevice*);

    private:
        int _sizeOfFrame;
        unsigned long long _socketBufferSize;
};

PELICAN_DECLARE_CHUNKER(EmbraceChunker)

} // namespace ampp
} // namespace pelican
#endif // EMBRACECHUNKER_H 
