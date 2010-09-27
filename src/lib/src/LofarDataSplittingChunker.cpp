#include "LofarDataSplittingChunker.h"

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
 * Constructor
 *
 * XML options.
 * ================================
 *
 * Read from base class:
 *  - data (type)  LIST.
 *  - connection (host, port)
 *
 * Class Specific options:
 *  - samplesPerPacket (value)
 *  - subbandsPerPacket (value)
 *  - nRawPolarisations (value)
 *
 *  - udpPacketsPerIteration (value)
 *  - clock (value)
 *  - dataBitSize (value)
 *
 */
LofarDataSplittingChunker::LofarDataSplittingChunker(const ConfigNode& config)
: AbstractChunker(config)
{
    // Check the configuration type matches the class name.
    if (config.type() != "LofarDataSplittingChunker")
        throw _err("LofarDataSplittingChunker(): "
                "Invalid or missing XML configuration.");

    // Retrieve configuration options.
    // -------------------------------------------------------------------------
    // Packet dimensions.
    _nSamples = config.getOption("samplesPerPacket", "value").toUInt();
    _nSubbands = config.getOption("subbandsPerPacket", "value").toUInt();
    _nPolarisations = config.getOption("nRawPolarisations", "value").toUInt();
    // Number of UDP packets collected into one chunk (iteration of the pipeline).
    _nPackets = config.getOption("udpPacketsPerIteration", "value").toUInt();
    // Clock => sample rate.
    _clock = config.getOption("clock", "value").toUInt();

    // Calculate the packet data size.
    size_t headerSize = sizeof(struct UDPPacket::Header);
    _packetSize = _nSubbands * _nSamples * _nPolarisations;
    unsigned sampleBits = config.getOption("dataBitSize", "value").toUInt();
    switch (sampleBits)
    {
        case 8:
            _packetSize = _packetSize * sizeof(TYPES::i8complex) + headerSize;
            break;
        case 16:
            _packetSize = _packetSize * sizeof(TYPES::i16complex) + headerSize;
            break;
        default:
            throw _err("LofarDataSplittingChunker(): "
                    "Unsupported number of data bits.");
    }

    // Initialise class variables.
    _startTime = _startBlockid = 0;
    _packetsAccepted = 0;
    _packetsRejected = 0;

    // Check a number of data chunk types are registerd to be written.
    // These are set in the XML.
    if (type().isEmpty())
        throw _err("LofarDataSplittingChunker(): Data type unspecified.");
}


/**
 * @details
 * Constructs a new QIODevice (in this case a QUdpSocket) and returns it
 * after binding the socket to the port specified in the XML node and read by
 * the constructor of the abstract chunker.
 */
QIODevice* LofarDataSplittingChunker::newDevice()
{
    QUdpSocket* socket = new QUdpSocket;

    if (!socket->bind(port()))
        cerr << "LofarDataSplittingChunker::newDevice(): "
                "Unable to bind to UDP port!" << endl;

    return socket;
}


/**
 * @details
 * Gets the next chunk of data from the UDP socket (if it exists).
 */
void LofarDataSplittingChunker::next(QIODevice* device)
{
    QUdpSocket* socket = static_cast<QUdpSocket*>(device);

    unsigned offset = 0;
    unsigned prevSeqid = _startTime;
    unsigned prevBlockid = _startBlockid;
    UDPPacket currPacket, emptyPacket;

    WritableData writableData1 = getDataStorage(_nPackets * _packetSize,
            "LofarChunkData1");
    WritableData writableData2 = getDataStorage(_nPackets * _packetSize,
            "LofarChunkData2");

    if (writableData1.isValid() && writableData2.isValid()) {

        // Loop over UDP packets.
        for (unsigned i = 0; i < _nPackets; ++i) {

            // Chunker sanity check.
            if (!isActive()) return;

            // Wait for datagram to be available.
            while (!socket->hasPendingDatagrams())
                socket->waitForReadyRead(100);

            if (socket->readDatagram(reinterpret_cast<char*>(&currPacket), _packetSize) <= 0) {
                cout << "LofarDataSplittingChunker::next(): "
                        "Error while receiving UDP Packet!" << endl;
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

            if (diff < _nSamples) { // Duplicated packets... ignore
                ++_packetsRejected;
                i -= 1;
                continue;
            }
            else if (diff > _nSamples) // Missing packets
                lostPackets = (diff / _nSamples) - 1; // -1 since it includes this includes the received packet as well

            if (lostPackets > 0) {
                printf("Generate %u empty packets, prevSeq: %u, new Seq: %u, prevBlock: %u, newBlock: %u\n",
                        lostPackets, prevSeqid, seqid, prevBlockid, blockid);
            }

            // Generate lostPackets empty packets, if any
            unsigned packetCounter = 0;
            for (packetCounter = 0; packetCounter < lostPackets && i + packetCounter < _nPackets; ++packetCounter)
            {
                // Generate empty packet with correct seqid and blockid
                prevSeqid = (prevBlockid + _nSamples < totBlocks) ? prevSeqid : prevSeqid + 1;
                prevBlockid = (prevBlockid + _nSamples) % totBlocks;
                generateEmptyPacket(emptyPacket, prevSeqid, prevBlockid);
                offset = writePacket(&writableData1, emptyPacket, offset);

                // Check if the number of required packets is reached
            }

            i += packetCounter;

            // Write received packet
            // FIXME: Packet will be lost if we fill up the buffer with sufficient empty packets...
            if (i != _nPackets) {
                ++_packetsAccepted;
                offset = writePacket(&writableData1, currPacket, offset);
                prevSeqid = seqid;
                prevBlockid = blockid;
            }
        }
    }
    else {
        // Must discard the datagram if there is no available space.
        socket->readDatagram(0, 0);
        cout << "LofarDataSplittingChunker::next(): "
                "Writable data not valid, discarding packets." << endl;
    }

//    // Update _startTime
//    _startTime = prevSeqid;
//    _startBlockid = prevBlockid;
}


/**
 * @details
 * Generates an empty UDP packet with no time stamp.
 * TODO: packet data generation can be done once!
 */
void LofarDataSplittingChunker::generateEmptyPacket(UDPPacket& packet, unsigned int seqid, unsigned int blockid)
{
    size_t size = _packetSize - sizeof(struct UDPPacket::Header);
    memset((void*) packet.data, 0, size);
    packet.header.nrBeamlets = _nSubbands;
    packet.header.nrBlocks   = _nSamples;
    packet.header.timestamp  = seqid;
    packet.header.blockSequenceNumber    = blockid;
}


/**
 * @details
 * Write packet to WritableData object
 */
int LofarDataSplittingChunker::writePacket(WritableData *writer, UDPPacket& packet, unsigned offset)
{
    if (writer->isValid()) {
        writer->write(reinterpret_cast<void*>(&packet), _packetSize, offset);
        return offset + _packetSize;
    }
    else {
        cerr << "LofarDataSplittingChunker::writePacket(): "
                "WritableData is not valid!" << endl;
        return -1;
    }
}

} // namespace lofar
} // namespace pelican
