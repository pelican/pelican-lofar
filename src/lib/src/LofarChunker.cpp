#include "LofarChunker.h"
#include "LofarUdpHeader.h"
#include "LofarTypes.h"
#include <QUdpSocket>
#include <stdio.h>

#include <iostream>

#include "pelican/utility/memCheck.h"

namespace pelican {
namespace lofar {

//PELICAN_DECLARE_CHUNKER(LofarChunker)

/**
 * @details
 * Constructs a new LofarChunker.
 */
LofarChunker::LofarChunker(const ConfigNode& config) : AbstractChunker(config)
{
    // TODO make configurable
    _nPackets = 1;
    _packetsAccepted = 0;
    _packetsRejected = 0;
    _samplesPerPacket = 0;
    _startTime;
}

/**
 * @details
 * Constructs a new device.
 */
QIODevice* LofarChunker::newDevice()
{
    QUdpSocket* socket = new QUdpSocket;
    QHostAddress hostAddress(host());
    socket->bind( hostAddress, port() );
    return socket;
}

/**
 * @details
 * Gets the next chunk of data from the UDP socket (if it exists).
 */
void LofarChunker::next(QIODevice* device)
{
    QUdpSocket *socket = static_cast<QUdpSocket*>(device);

    int packetSize = sizeof(UDPPacket);
    size_t offset = 0;
    UDPPacket currPacket;
    UDPPacket emptyPacket;
    generateEmptyPacket(emptyPacket);
    unsigned    previousSeqid       = 0;
    TYPES::TimeStamp   actualStamp  = _samplesPerPacket;

    std::cout << "packetSize: " << packetSize << ", nPackets: " << _nPackets << std::endl;
    WritableData writableData = getDataStorage(_nPackets * packetSize);
    if (! writableData.isValid())
        throw QString("LofarChunker::next(): Writable data not valid.");

    // Loop over UDP packets.
    for (int i = 0; i < _nPackets; i++) {

        qint64 sizeDatagram;
        
        // Interruptible read, to allow stopping this thread even if the station does not send data
        std::cout << "LofarChunker::next(): Waiting for ready read." << std::endl;
        socket -> waitForReadyRead();
        if ( ( sizeDatagram = socket->readDatagram(reinterpret_cast<char*>(&currPacket), packetSize) ) <= 0 ) {
            printf("LofarChunker::next(): Error while receiving UDP Packet: %d\n", (int) sizeDatagram);
            continue;
        }

        // TODO Check for endianness
        ++_packetsAccepted;
        unsigned seqid   = currPacket.header.timestamp;
        unsigned blockid = currPacket.header.blockSequenceNumber;

        // If the seconds counter is 0xFFFFFFFF, the data cannot be trusted
        if (seqid == ~0U) {
            ++_packetsRejected;
            writableData.write(reinterpret_cast<void*>(&emptyPacket), packetSize, offset);
            offset += packetSize;
            continue;
        }

        // Check that the packets are contiguous
        if (previousSeqid + 1 != seqid) {
            unsigned lostPackets = seqid - previousSeqid;

            // Generate lostPackets empty packets
            for (unsigned packetCounter = 0; packetCounter < lostPackets; packetCounter++) {
                writableData.write(reinterpret_cast<void*>(&emptyPacket), packetSize, offset);
                offset += packetSize;
            }

            i += lostPackets;
            previousSeqid = seqid + lostPackets;
            continue;
        }


        previousSeqid = seqid;
        if (i == 0)
            actualStamp.setStamp(seqid, blockid);

        // Write the data.
        writableData.write(reinterpret_cast<void*>(&currPacket), packetSize, offset);

        offset += packetSize;
    }

}

/**
 * @details
 * Generates an empty UDP packet with no time stamp.
 */
void LofarChunker::generateEmptyPacket(UDPPacket& packet)
{
    size_t size = sizeof(packet.data);
    memset((void*)packet.data, 0, size);
    packet.header.nrBlocks = 0;
    packet.header.timestamp = 0;
    packet.header.blockSequenceNumber = 0;
}

} // namespace lofar
} // namespace pelican
