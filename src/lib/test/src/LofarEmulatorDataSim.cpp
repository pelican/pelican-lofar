#include "LofarEmulatorDataSim.h"
#include "LofarUdpHeader.h"
#include "LofarTypes.h"

#include "pelican/utility/ConfigNode.h"

#include <iostream>
using std::cout;
using std::endl;
#include <cmath>

namespace pelican {
namespace lofar {

/**
 * @details Constructs the LOFAR UDP emulator.
 */
LofarEmulatorDataSim::LofarEmulatorDataSim(const ConfigNode& configNode)
    : AbstractUdpEmulator(configNode)
{
    // Get the config.
    _interval = configNode.getOption("packetSendInterval", "value", "100").toULong();
    _startDelay = configNode.getOption("packetStartDelay", "value", "1").toInt();

    _nSubbands = configNode.getOption("subbandsPerPacket", "value", "62").toInt();
    _nPolarisations = configNode.getOption("polsPerPacket", "value", "2").toInt();

    // Fixed parameters.
    _samplesPerPacket = 16;

    _clock = 200;
    _blockid = _samplesPerPacket; // blockid offset
    _nPackets = -1; // Continue for ever.

    _timestamp = 1;
    _packetCounter = 0;
    _packetSize = sizeof(UDPPacket);

    _setPacketHeader();
}




/**
 * @details
 * Returns a packet of LOFAR-style data.
 *
 * @param ptr[out]   Pointer to the start of the packet.
 * @param size[out]  Size of the packet.
 */
void LofarEmulatorDataSim::getPacketData(char*& ptr, unsigned long& size)
{
    // Fill the packet
    _setPacketData();

    // An interval consists of 1 second of data. The timestamp represents the
    // integer value for the integer. Each interval consists of a number of
    // blocks no_blocks, which is defined in the header.
    // The total number of blocks is clockspeed / no_blocks.
    _packet.header.timestamp = 1 + (_blockid + _samplesPerPacket) / (_clock == 160 ? 156250 : (_timestamp % 2 == 0 ? 195313 : 195212));
    _packet.header.blockSequenceNumber = (_blockid + _samplesPerPacket) % (_clock == 160 ? 156250 : (_timestamp % 2 == 0 ? 195313 : 195212));
    _timestamp = _packet.header.timestamp;
    _blockid = _packet.header.blockSequenceNumber;

    // Return values.
    ptr = (char*)(&_packet);
    size = _packetSize;

    // Increment packet counter.
    _packetCounter++;
}




/**
* @details
* Sets the UDP packet header that will be used.
*/
void LofarEmulatorDataSim::_setPacketHeader()
{
    _packet.header.version    = (uint8_t) 0xAAAB;
    _packet.header.nrBeamlets = _nSubbands;
    _packet.header.nrBlocks   = _samplesPerPacket;
    _packet.header.station    = (uint16_t) 0x00EA;
    _packet.header.sourceInfo = (uint8_t) 1010;
}




/**
 * @details
 * Fills the packet with data.
 */
void LofarEmulatorDataSim::_setPacketData()
{
    // Calculate the actual required packet size.
    _packetSize = _nSubbands * _samplesPerPacket * _nPolarisations;
    _packetSize *= sizeof(Complex16);
    _packetSize += sizeof(struct UDPPacket::Header);

    // Check that data is not too large.
    if (_packetSize > sizeof(_packet.data))
    {
        throw QString("LofarEmulatorSimData: "
                "Packet size (%1) too large (max %2).").
                arg(_packetSize).arg(sizeof(_packet.data));
    }

    unsigned i;
    int16 re, im;
    unsigned long time0 =  _packetCounter * _samplesPerPacket;
    float sampleRate = 122.0f;
    float f0 = 0.0f;
    float time;
    float freq;
    float arg;
    float twoPi = 3.14159265358979323846264338327950 * 2.0f;

    Complex16 *data = reinterpret_cast<Complex16*>(_packet.data);
    for (int s = 0; s < _nSubbands; s++)
    {
        for (int t = 0; t < _samplesPerPacket; t++)
        {
            i = _nPolarisations * (t + s * _samplesPerPacket);

            time = float(time0 + t) / sampleRate;

            // polarisation 1.
            freq = f0 + float(s*2);
            arg = twoPi * freq * time;
            re = 20.0f * cos(arg);
            im = time0 + t;//freq; // 20.0f * sin(arg);
            data[i + 0] = Complex16(re, im);

            // polarisation 2.
            freq = f0 + float(s*2) + 1.0f;
            arg = twoPi * freq * time;
            re = (time0 + t); //30.0f * cos(arg);
            im = 30.0f * sin(arg);

            data[i + 1] = Complex16(re, im);
        }
    }
}





} // namespace lofar
} // namespace pelican
