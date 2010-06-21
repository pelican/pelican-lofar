#ifndef LOFARCHUNKER_H
#define LOFARCHUNKER_H
#include <QtCore/QString>
#include <QtCore/QObject>
#include "LofarTypes.h"
#include "LofarUdpHeader.h"
#include "pelican/server/AbstractChunker.h"

/**
 * @file LofarChunker.h
 */

namespace pelican {
namespace lofar {

class DataManager;

/**
 * @class LofarChunker
 *
 * @brief
 * Implementation of an AbstractChunker to monitor calling.
 *
 * @details
 *
 */
class LofarChunker : public AbstractChunker
{

    public:
        /// Constructs a new LofarChunker.
        LofarChunker(const ConfigNode&);

        /// Destroys the LofarChunker.
        ~LofarChunker() { } 

        /// Creates the socket to use for the incoming data stream.
        virtual QIODevice* newDevice();

        ///
        virtual void next(QIODevice*);

        /// Sets the number of packets to read.
        void setPackets(int packets) {_nPackets = packets;}

    private:
        /// Generates an empty UDP packet.
        void generateEmptyPacket(UDPPacket& packet, unsigned int seqid, unsigned int blockid);

        /// Write UDPPacket to writeableData object
        unsigned writePacket(WritableData *writer, UDPPacket& packet, unsigned offset);

    private:
  
        int _nPackets;
        unsigned _packetsRejected;
        unsigned _packetsAccepted;
        unsigned _samplesPerPacket;
        unsigned _subbandsPerPacket;
        unsigned _nrPolarisations;
        unsigned _startTime;
        unsigned _startBlockid;
        unsigned _packetSize;
        unsigned _clock;

        friend class LofarChunkerTest;
};

} // namespace lofar
} // namespace pelican

#endif // LOFARCHUNKER_H
