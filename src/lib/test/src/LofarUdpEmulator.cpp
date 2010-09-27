#include "LofarUdpEmulator.h"
#include "LofarUdpHeader.h"
#include "LofarTypes.h"

#include "pelican/utility/ConfigNode.h"

#include <cmath>

namespace pelican {
namespace lofar {

/**
 * @details
 * Constructs the LOFAR UDP emulator.
 */
LofarUdpEmulator::LofarUdpEmulator(const ConfigNode& configNode)
: AbstractUdpEmulator(configNode)
{
    // Load data (packet) parameters from the configuration.
    _subbandsPerPacket = configNode.getOption("packet", "subbands", "1").toInt();
    _samplesPerPacket = configNode.getOption("packet", "samples", "1").toInt();
    _nrPolarisations = configNode.getOption("packet", "polarisations", "2").toInt();
    _clock = configNode.getOption("params", "clock").toInt();
    int sampleSize = configNode.getOption("packet", "sampleSize", "8").toInt();

    // Load Emulator parameters. (Interval = microseconds between sending packets)
    _interval = configNode.getOption("packet", "interval", "100000").toULong();
    _startDelay = configNode.getOption("packet", "startDelay", "0").toInt();
    _nPackets = configNode.getOption("packet", "nPackets", "-1").toInt();

    // Setup emulator class variables.
    switch (sampleSize)
    {
        case 8: { _sampleType = i8complex;  break; }
        case 16:{ _sampleType = i16complex; break; }
        default: throw QString("LofarUdpEmulator: Unsupported sample size.");
    }
    _blockid = _samplesPerPacket; // blockid offset
    _timestamp = 1;
    _packetCounter = 0;
    _packetSize = sizeof(UDPPacket);

    // Fill the packet.
    setPacketHeader(0);
    fillPacket();
}


/**
 * @details
 * Fills the packet with data.
 */
void LofarUdpEmulator::fillPacket()
{
    typedef TYPES::i8complex i8c;
    typedef TYPES::i16complex i16c;

    // Packet data is generated once and all packets will contain the same data
    // records. Each block consists of a time slice (one sample) for each
    // beamlet in beamlet order.

    // Calculate the actual required packet size.
    _packetSize = _subbandsPerPacket * _samplesPerPacket * _nrPolarisations;
    switch (_sampleType) {
        case i8complex: { _packetSize *= sizeof(i8c);  break; }
        case i16complex:{ _packetSize *= sizeof(i16c); break; }
    }
    _packetSize += sizeof(struct UDPPacket::Header);


    // Check that packet size is not too large.
    if (_packetSize > sizeof(_packet.data))
        throw QString("LofarUdpEmulator: Packet size (%1) too large (max %2).")
            .arg(_packetSize).arg(sizeof(_packet.data));



    // Create test data in packet.
    unsigned idx;
    switch (_sampleType)
    {
        case i8complex:
        {
            i8c* samples = reinterpret_cast<i8c*>(_packet.data);

            for (int i = 0; i < _samplesPerPacket; i++)
            {
                for (int j = 0; j < _subbandsPerPacket; j++)
                {
                    idx = _nrPolarisations * (j + i * _subbandsPerPacket);

                    samples[idx] = i8c(i + j, i);
                    samples[idx + 1] = i8c(i + j, j);
                }
            }
            break;
        }
        case i16complex:
        {
            i16c *s = reinterpret_cast<i16c*>(_packet.data);
            for (int i = 0; i < _samplesPerPacket; i++)
            {
                for (int j = 0; j < _subbandsPerPacket; j++)
                {
                    idx = _nrPolarisations * (j + i * _subbandsPerPacket);
                    s[idx] = i16c(i + j, i);
                    s[idx + 1] = i16c(i + j, j);
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
    /* An interval consists of 1 second of data. The timestamp represents the integer
       value for the integer. Each interval consists of a number of blocks no_blocks,
       which is defined in the header. The total number of blocks is clockspeed/no_blocks.
    */

    // Set the return variables.
    size = _packetSize;
    ptr = (char*)(&_packet);

    // Calculate seqid and blockid from packet counter and clock
    if (!_looseEvenPackets || (_looseEvenPackets && _packetCounter % 2 == 1)) {
        _packet.header.timestamp = 1 + (_blockid + _samplesPerPacket) /
                (_clock == 160 ? 156250 : (_timestamp % 2 == 0 ? 195313 : 195212));
        _packet.header.blockSequenceNumber = (_blockid + _samplesPerPacket) %
                (_clock == 160 ? 156250 : (_timestamp % 2 == 0 ? 195313 : 195212));
        _timestamp = _packet.header.timestamp;
        _blockid = _packet.header.blockSequenceNumber;
    }
    else {
        _packet.header.timestamp = _timestamp;
        _packet.header.blockSequenceNumber = _blockid;
        _timestamp = 1 + (_blockid + _samplesPerPacket) / (_clock == 160 ? 156250 : (_timestamp % 2 == 0 ? 195313 : 195212));
        _blockid = (_blockid + _samplesPerPacket) %
                (_clock == 160 ? 156250 : (_timestamp % 2 == 0 ? 195313 : 195212));
   }

    //fillPacket();

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
    if (!packetHeader)
    {
        _packet.header.version    = (uint8_t) 0xAAAB;   // Hardcoded in RSP firmware
        _packet.header.nrBeamlets = _subbandsPerPacket;
        _packet.header.nrBlocks   = _samplesPerPacket;
        _packet.header.station    = (uint16_t) 0x00EA;  // Pelican
        _packet.header.sourceInfo = (uint8_t) 1010;     // LofarUdpEmulator!
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
void LofarUdpEmulator::looseEvenPackets(bool loose)
{
    _looseEvenPackets = loose;
}

} // namespace lofar
} // namespace pelican
