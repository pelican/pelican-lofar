#include "ABEmulator.h"
#include "pelican/utility/ConfigNode.h"
#include <cmath>

namespace pelican {
namespace ampp {

/*
 * Constructs the ABEmulator.
 * This obtains the relevant configuration parameters.
 */
ABEmulator::ABEmulator(const ConfigNode& configNode)
    : AbstractUdpEmulator(configNode)
{
    // Initialise defaults.
    _counter = 0;
    _totalSamples = 0;
    _samples = configNode.getOption("packet", "samples", "1024").toULong();
    _interval = configNode.getOption("packet", "interval",
            QString::number(_samples * 10)).toULong(); // Interval in micro-sec.
    _period = configNode.getOption("signal", "period", "20").toULong();
    _omega = (2.0 * 3.14159265) / _period; // Angular frequency.

    // Set the packet size in bytes (each sample is 8 bytes + 8 for header
    // + 8 for (ignored) footer).
    _packet.resize(_samples * 8 + 8 + 8);

    // Set constant parts of packet header data.
    char* ptr = _packet.data();
    // Packet counter.
    *reinterpret_cast<short int*>(ptr + 0) = (short int) ((_counter & 0x00FF0000) >> 32);
    *reinterpret_cast<int*>(reinterpret_cast<short int*>(ptr + 1)) = (int) (_counter & 0x0000FFFF);
    *(ptr + 6) = _specQuart; // Spectral quarter.
    *(ptr + 7) = _beam; // Beam number.
}

/*
 * Creates a packet of UDP signal data containing the psuedo-Stokes of the FFT
 * of a sine wave, setting the pointer to the start of the packet, and the size
 * of the packet.
 */
void ABEmulator::getPacketData(char*& ptr, unsigned long& size)
{
    // Set pointer to the output data.
    ptr = _packet.data();
    size = _packet.size();

    // Set the packet header.
    *reinterpret_cast<short int*>(ptr + 0) = (short int) ((_counter & 0x00FF0000) >> 32);
    *reinterpret_cast<int*>(reinterpret_cast<short int*>(ptr + 1)) = (int) (_counter & 0x0000FFFF);

    // Fill the packet data.
    char* data = ptr + 8; // Add offset for header.
    for (unsigned i = 0; i < _samples; ++i) {
        //float value = sin(((_totalSamples + i) % _period) * _omega);
        short int XXre = i * 4 + 0;
        short int YYre = i * 4 + 1;
        short int XYre = i * 4 + 2;
        short int XYim = i * 4 + 3;
        reinterpret_cast<short int*>(data)[i * 4 + 0] = XXre;
        reinterpret_cast<short int*>(data)[i * 4 + 1] = YYre;
        reinterpret_cast<short int*>(data)[i * 4 + 2] = XYre;
        reinterpret_cast<short int*>(data)[i * 4 + 3] = XYim;
    }

    // Increment counters for next time.
    _counter++;
    _totalSamples += _samples;
}

} // namespace ampp
} // namespace pelican

