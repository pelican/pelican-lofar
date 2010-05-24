#include "LofarUdpEmulator.h"
#include "LofarUdpHeader.h"
#include "LofarTypes.h"

#include "pelican/utility/ConfigNode.h"
#include "pelican/utility/memCheck.h"

namespace pelican {
namespace lofar {

/**
 * @details Constructs the LOFAR UDP emulator.
 */
LofarUdpEmulator::LofarUdpEmulator(const ConfigNode& configNode)
    : AbstractUdpEmulator(configNode)
{
    _packetCounter = 0;
    _packetSize = sizeof(UDPPacket);
    _interval = configNode.getOption("packet", "interval", "100000").toULong();

    // Set data parameters.
    _subbandsPerPacket = configNode.getOption("packet", "subbands", "1").toInt();
    _samplesPerPacket = configNode.getOption("packet", "samples", "1").toInt();
    _nrPolarisations = configNode.getOption("packet", "polarisations", "2").toInt();

    // Set test parameters.
    int sampleSize = configNode.getOption("packet", "sampleSize", "8").toInt();
    if (sampleSize == 4) _sampleType = i4complex;
    else if (sampleSize == 16) _sampleType = i16complex;
    else _sampleType = i8complex;
    _nPackets = configNode.getOption("packet", "nPackets", "-1").toInt();
    _startDelay = configNode.getOption("packet", "startDelay", "0").toInt();
    _seqNumbers = 0;

    // Fill the packet.
    setPacketHeader(NULL);
    fillPacket();
}

/**
 * @details
 * Fills the packet with data.
 */
void LofarUdpEmulator::fillPacket()
{
    // Calculate the actual required packet size.
    _packetSize = _subbandsPerPacket * _samplesPerPacket * _nrPolarisations;
    switch (_sampleType) {
        case i4complex: _packetSize *= sizeof(TYPES::i4complex); break;
        case i8complex: _packetSize *= sizeof(TYPES::i8complex); break;
        case i16complex: _packetSize *= sizeof(TYPES::i16complex); break;
    }
    _packetSize += sizeof(struct UDPPacket::Header);

    // Check that packet size is not too large.
    if (_packetSize > sizeof(_packet.data))
        throw QString("LofarUdpEmulator: Packet size (%1) too large (max %2).")
            .arg(_packetSize).arg(sizeof(_packet.data));

    // Create test data in packet.
    switch (_sampleType) {
    case i4complex: {
        TYPES::i4complex *s = reinterpret_cast<TYPES::i4complex*>(_packet.data);
        for (int i = 0; i < _samplesPerPacket; i++) {
            for (int j = 0; j < _subbandsPerPacket; j++) {
                unsigned index = i * _subbandsPerPacket * _nrPolarisations +
                        j * _nrPolarisations;
                s[index] = TYPES::i4complex(i + j, i);
                s[index + 1] = TYPES::i4complex(i + j, i);
            }
        }
        break;
    }
    case i8complex: {
        TYPES::i8complex *s = reinterpret_cast<TYPES::i8complex*>(_packet.data);
        for (int i = 0; i < _samplesPerPacket; i++) {
            for (int j = 0; j < _subbandsPerPacket; j++) {
                unsigned index = i * _subbandsPerPacket * _nrPolarisations +
                        j * _nrPolarisations;
                s[index] = TYPES::i8complex(i + j, i);
                s[index + 1] = TYPES::i8complex(i + j, i);
            }
        }
        break;
    }
    case i16complex: {
        TYPES::i16complex *s = reinterpret_cast<TYPES::i16complex*>(_packet.data);
        for (int i = 0; i < _samplesPerPacket; i++) {
            for (int j = 0; j < _subbandsPerPacket; j++) {
                unsigned index = i * _subbandsPerPacket * _nrPolarisations +
                        j * _nrPolarisations;
                s[index] = TYPES::i16complex(i + j, i);
                s[index + 1] = TYPES::i16complex(i + j, i);
            }
        }
        break;
    }
    }
}

/**
 * @details
 * Returns a packet of LOFAR-style data.
 *
 * @param ptr[out]   Pointer to the start of the packet.
 * @param size[out]  Size of the packet.
 */
void LofarUdpEmulator::getPacketData(char*& ptr, unsigned long& size)
{
    // Set the return variables.
    size = _packetSize;
    ptr = (char*)(&_packet);

    // Update packet timestamp.
    if (_seqNumbers == NULL) {
        _packet.header.timestamp = _packetCounter;
        _packet.header.blockSequenceNumber = _packetCounter;
    } else {
        _packet.header.timestamp = _seqNumbers[_packetCounter];
        _packet.header.blockSequenceNumber = _seqNumbers[_packetCounter];
    }

    // Increment packet counter.
    _packetCounter++;
}

/**
* @details
* Sets the UDP packet header that will be used.
*
* @param packetHeader Pointer to memory containing the packet header.
*/
void LofarUdpEmulator::setPacketHeader(UDPPacket::Header* packetHeader)
{
    // Create packet header.
    if (packetHeader == NULL) {
        _packet.header.version    = 1;
        _packet.header.nrBeamlets = 1;
        _packet.header.nrBlocks   = 1;
        _packet.header.station    = 1;
        _packet.header.sourceInfo = 1;
    }
    else {
        // Copy header to packet.
        memcpy(&(_packet.header), packetHeader, sizeof(UDPPacket::Header));
    }
}

/**
 * @details
 * Sets the sequence numbers.
 */
void LofarUdpEmulator::setSeqNumbers(unsigned int *seqNumbers)
{
    _seqNumbers = seqNumbers;
}

} // namespace lofar
} // namespace pelican
