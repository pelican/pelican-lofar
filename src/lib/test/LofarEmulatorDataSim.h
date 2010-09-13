#ifndef LOFAR_EMULATOR_DATA_SIM_H
#define LOFAR_EMULATOR_DATA_SIM_H

#include "pelican/emulator/AbstractUdpEmulator.h"
#include "LofarUdpHeader.h"
#include "LofarTypes.h"

/**
 * @file LofarEmulatorDataSim.h
 */

namespace pelican {

class ConfigNode;

namespace lofar {

/**
 * @class LofarEmulatorDataSim
 *
 * @brief
 *
 * @details
 *
 */
class LofarEmulatorDataSim : public AbstractUdpEmulator
{
    public:
        typedef TYPES::i16complex Complex16;
        typedef TYPES::int16 int16;

    public:
        /// Constructor.
        LofarEmulatorDataSim(const ConfigNode& configNode);

        /// Destructor.
        ~LofarEmulatorDataSim() {}

    public: // override of virtual functions from the base class.
        /// Returns a UDP packet.
        void getPacketData(char*& ptr, unsigned long& size);

        /// Returns the interval between packets in microseconds.
        unsigned long interval() { return _interval; }

        /// Returns the number of packets to send. Runs forever if negative.
        int nPackets() { return _nPackets; }

        /// Returns the start delay in seconds.
        int startDelay() {return _startDelay;}

    public:
        /// Sets the packet header.
        void _setPacketHeader();

        /// Fills the packet with data.
        void _setPacketData();


    private:
        UDPPacket _packet; // The packet to send.

        int _nSubbands; // == Number of beamlets
        int _nPolarisations;
        int _samplesPerPacket;

        unsigned _interval;   // microseconds.
        unsigned _startDelay; // seconds.
        int _nPackets;        // number of packets to send -1 = contineu forever.

        unsigned long _packetCounter;
        unsigned long _packetSize;

        unsigned _clock;     // Station clock speed
        unsigned _timestamp; // The time-stamp of the preceding packet
        unsigned _blockid;   // The blockSequenceNumber of the preceding packet
};


} // namespace lofar
} // namespace pelican
#endif // LOFAR_EMULATOR_DATA_SIM_H
