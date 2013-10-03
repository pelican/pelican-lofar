#include "LofarChunker.h"
#include "LofarUdpHeader.h"
#include "LofarTypes.h"

#include <QtNetwork/QUdpSocket>

#include <cstdio>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

namespace pelican {
namespace lofar {

/**
 * @details
 * Constructs a new LofarChunker.
 *
 * TODO: this assumes variable packet size. make this a configuration option.
 */
LofarChunker::LofarChunker(const ConfigNode& config) : AbstractChunker(config)
{
    if (config.type() != "LofarChunker")
        throw QString("LofarChunker::LofarChunker(): Invalid configuration");

    // Get configuration options
    int _sampleType = config.getOption("dataBitSize", "value").toInt();
    _samplesPerPacket = config.getOption("samplesPerPacket", "value").toInt();
    _subbandsPerPacket = config.getOption("subbandsPerPacket", "value").toInt();
    _nrPolarisations = config.getOption("nRawPolarisations", "value").toInt();
    _clock = config.getOption("clock", "value").toInt();
    _startTime = _startBlockid = 0;
    _packetsAccepted = 0;
    _packetsRejected = 0;

    // Calculate the number of ethernet frames that will go into a chunk
    _nPackets = config.getOption("udpPacketsPerIteration", "value").toUInt();

    // Some sanity checking.
    if (config.type().isEmpty())
        throw QString("LofarChunker::LofarChunker(): Data type unspecified.");

    _packetSize = _subbandsPerPacket * _samplesPerPacket * _nrPolarisations;

    size_t headerSize = sizeof(struct UDPPacket::Header);

    switch (_sampleType)
    {
        case 4:
            _packetSize = _packetSize * sizeof(TYPES::i4complex) + headerSize;
            break;
        case 8:
            _packetSize = _packetSize * sizeof(TYPES::i8complex) + headerSize;
            break;
        case 16:
            _packetSize = _packetSize * sizeof(TYPES::i16complex) + headerSize;
            break;
    }
}


/**
 * @details
 * Constructs a new QIODevice (in this case a QUdpSocket) and returns it
 * after binding the socket to the port specified in the XML node and read by
 * the constructor of the abstract chunker.
 */
QIODevice* LofarChunker::newDevice()
{
    QUdpSocket* socket = new QUdpSocket;

    if (!socket->bind(port()))
        cerr << "LofarChunker::newDevice(): Unable to bind to UDP port!" << endl;

    return socket;
}


/**
 * @details
 * Gets the next chunk of data from the UDP socket (if it exists).
 */
void LofarChunker::next(QIODevice* device)
{
    QUdpSocket* socket = static_cast<QUdpSocket*>(device);

    unsigned offset = 0;
    unsigned prevSeqid = _startTime;
    unsigned prevBlockid = _startBlockid;
    UDPPacket currPacket, emptyPacket;

    WritableData writableData = getDataStorage(_nPackets * _packetSize);

    if (writableData.isValid()) {

        // Loop over UDP packets.
        for (unsigned i = 0; i < _nPackets; ++i) {

            // Chunker sanity check.
            if (!isActive()) return;

            // Wait for datagram to be available.
            while (!socket -> hasPendingDatagrams())
                socket -> waitForReadyRead(100);

            if (socket->readDatagram(reinterpret_cast<char*>(&currPacket), _packetSize) <= 0) {
                cout << "LofarChunker::next(): Error while receiving UDP Packet!" << endl;
                i--;
                continue;
            }

            // Check for endianness. Packet data is in little endian format.
            unsigned seqid, blockid;

#if Q_BYTE_ORDER == Q_BIG_ENDIAN
            // TODO: Convert from little endian to big endian.
            seqid   = currPacket.header.timestamp;
            blockid = currPacket.header.blockSequenceNumber;
#elif Q_BYTE_ORDER == Q_LITTLE_ENDIAN
            seqid   = currPacket.header.timestamp;
            blockid = currPacket.header.blockSequenceNumber;
#endif

            // First time next has been run, initialise startTime and startBlockId
            if (i == 0 && _startTime == 0) {
                prevSeqid = _startTime = _startTime == 0 ? seqid : _startTime;
                prevBlockid = _startBlockid = _startBlockid == 0 ? blockid : _startBlockid;
            }

            // Sanity check in seqid. If the seconds counter is 0xFFFFFFFF,
            // the data cannot be trusted (ignore)
            if (seqid == ~0U || prevSeqid + 10 < seqid) {
                ++_packetsRejected;
                i -= 1;
                continue;
            }

            // Check that the packets are contiguous. Block id increments by no_blocks
            // which is defined in the header. Blockid is reset every interval (although
            // it might not start from 0 as the previous frame might contain data from this one)
            unsigned totBlocks = _clock == 160 ? 156250 : (prevSeqid % 2 == 0 ? 195313 : 195312);
            unsigned lostPackets = 0, diff = 0;

            diff =  (blockid >= prevBlockid) ? (blockid - prevBlockid) : (blockid + totBlocks - prevBlockid);

            if (diff < _samplesPerPacket) { // Duplicated packets... ignore
                ++_packetsRejected;
                i -= 1;
                continue;
            }
            else if (diff > _samplesPerPacket) // Missing packets
                lostPackets = (diff / _samplesPerPacket) - 1; // -1 since it includes this includes the received packet as well

            if (lostPackets > 0) {
                printf("Generate %u empty packets, prevSeq: %u, new Seq: %u, prevBlock: %u, newBlock: %u\n",
                        lostPackets, prevSeqid, seqid, prevBlockid, blockid);
            }

            // Generate lostPackets empty packets, if any
            unsigned packetCounter = 0;
            for (packetCounter = 0; packetCounter < lostPackets && i + packetCounter < _nPackets; ++packetCounter)
            {
                // Generate empty packet with correct seqid and blockid
                prevSeqid = (prevBlockid + _samplesPerPacket < totBlocks) ? prevSeqid : prevSeqid + 1;
                prevBlockid = (prevBlockid + _samplesPerPacket) % totBlocks;
                generateEmptyPacket(emptyPacket, prevSeqid, prevBlockid);
                offset = writePacket(&writableData, emptyPacket, offset);

                // Check if the number of required packets is reached
            }

            i += packetCounter;

            // Write received packet
            // FIXME: Packet will be lost if we fill up the buffer with sufficient empty packets...
            if (i != _nPackets) {
                ++_packetsAccepted;
                offset = writePacket(&writableData, currPacket, offset);
                prevSeqid = seqid;
                prevBlockid = blockid;
            }
        }
    }
    else {
        // Must discard the datagram if there is no available space.
        socket->readDatagram(0, 0);
        cout << "LofarChunker::LofarChunker(): "
                "Writable data not valid, discarding packets." << endl;
    }

    // Update _startTime
    _startTime = prevSeqid;
    _startBlockid = prevBlockid;
}


/**
 * @details
 * Generates an empty UDP packet with no time stamp.
 */
void LofarChunker::generateEmptyPacket(UDPPacket& packet, unsigned int seqid, unsigned int blockid)
{
    size_t size = _packetSize - sizeof(struct UDPPacket::Header);
    memset((void*) packet.data, 0, size);
    packet.header.nrBeamlets = _subbandsPerPacket;
    packet.header.nrBlocks   = _samplesPerPacket;
    packet.header.timestamp  = seqid;
    packet.header.blockSequenceNumber    = blockid;
}


/**
 * @details
 * Write packet to WritableData object
 */
int LofarChunker::writePacket(WritableData *writer, UDPPacket& packet, unsigned offset)
{
    if (writer->isValid()) {
        writer->write(reinterpret_cast<void*>(&packet), _packetSize, offset);
        return offset + _packetSize;
    }
    else {
        cerr << "LofarChunker::writePacket(): WritableData is not valid!" << endl;
        return -1;
    }
}


} // namespace lofar
} // namespace pelican
