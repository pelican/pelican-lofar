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
    if (config.type() != "LofarChunker")
        throw QString("LofarChunker::LofarChunker(): Invalid configuration");

    // Get configuration options
    int _sampleType = config.getOption("samples", "type").toInt();
    _samplesPerPacket = config.getOption("params","samplesPerPacket").toInt();
    _subbandsPerPacket = config.getOption("params","subbandsPerPacket").toInt();
    _nrPolarisations = config.getOption("params","nrPolarisation").toInt();
    _nPackets = config.getOption("params","nPackets").toInt();
    _packetsAccepted = 0;
    _packetsRejected = 0;
    _startTime = -1;

    // Some sanity checking.
    if (type().isEmpty())
        throw QString("TestUdpChunker::TestUdpChunker(): Data type unspecified.");

    _packetSize = _subbandsPerPacket * _samplesPerPacket * _nrPolarisations;

    switch (_sampleType) {
        case 4:  { _packetSize = _packetSize * sizeof(TYPES::i4complex)  + sizeof(struct UDPPacket::Header);  break; }
        case 8:  { _packetSize = _packetSize * sizeof(TYPES::i8complex)  + sizeof(struct UDPPacket::Header);  break; }
        case 16: { _packetSize = _packetSize * sizeof(TYPES::i16complex) + sizeof(struct UDPPacket::Header);  break; }
    }
}

/**
 * @details
 * Constructs a new device.
 */
QIODevice* LofarChunker::newDevice()
{
    QUdpSocket* socket = new QUdpSocket;
    QHostAddress hostAddress(host());
    socket -> bind( hostAddress, port() );
    return socket;
}

/**
 * @details
 * Gets the next chunk of data from the UDP socket (if it exists).
 */
void LofarChunker::next(QIODevice* device)
{
    QUdpSocket *socket = static_cast<QUdpSocket*>(device);

    unsigned         offset                    = 0;
    unsigned         previousSeqid             = _startTime;
    UDPPacket        currPacket, emptyPacket;
    qint64           sizeDatagram;

    WritableData writableData = getDataStorage(_nPackets * _packetSize);
    generateEmptyPacket(emptyPacket);

    if (! writableData.isValid())
        throw QString("LofarChunker::next(): Writable data not valid.");

    // Loop over UDP packets.
    for (int i = 0; i < _nPackets; i++) {
       
        // Interruptible read, to allow stopping this thread even if the station does not send data
        socket -> waitForReadyRead();
        if ( ( sizeDatagram = socket->readDatagram(reinterpret_cast<char*>(&currPacket), _packetSize) ) <= 0 ) {
            std::cout << "LofarChunker::next(): Error while receiving UDP Packet!" << std::endl;
            i--;
            continue;
        }

        // TODO Check for endianness
        ++_packetsAccepted;
        unsigned seqid   = currPacket.header.timestamp;
        unsigned blockid = currPacket.header.blockSequenceNumber;

        // First time next has been run, initialise startTime
        if (i == 0 && _startTime == 0)
            _startTime = previousSeqid = seqid;

        // If the seconds counter is 0xFFFFFFFF, the data cannot be trusted
        if (seqid == ~0U) {
            ++_packetsRejected;
            offset = writePacket(&writableData, emptyPacket, offset);
            continue;
        }

        // Check if packet seqid is smaller than expected (due to duplicates
        // or out-of-order packet arrivals. If so ignore
        if (previousSeqid + 1 > seqid) {
            i -= 1;
            continue;
        }

        // Check that the packets are contiguous
        if (previousSeqid + 1 != seqid) {
            unsigned lostPackets = seqid - previousSeqid - 1;

            // Generate lostPackets empty packets
            for (unsigned packetCounter = 0; packetCounter < lostPackets && 
                                             i + packetCounter < _nPackets; packetCounter++) {
                offset = writePacket(&writableData, emptyPacket, offset);
                i += 1;
            }

            // Write received packet
            offset = writePacket(&writableData, currPacket, offset);
            previousSeqid = seqid;
            continue;
        }

        // Write the data.
        offset = writePacket(&writableData, currPacket, offset);
        previousSeqid = seqid;
    }
 
    // Update _startTime
    _startTime = previousSeqid;
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

/**
 * @details
 * Generates an empty UDP packet with no time stamp.
 */
unsigned LofarChunker::writePacket(WritableData *writer, UDPPacket& packet, unsigned offset)
{
    if (writer -> isValid()) {
        writer -> write(reinterpret_cast<void*>(&packet), _packetSize, offset);
        return offset + _packetSize;
    } else {
        printf("WriteableData is not valid!\n");
        return -1;
    }
}

} // namespace lofar
} // namespace pelican
