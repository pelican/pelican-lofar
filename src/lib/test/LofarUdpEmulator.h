#ifndef LOFARUDPEMULATOR_H
#define LOFARUDPEMULATOR_H

#include "pelican/testutils/AbstractUdpEmulator.h"
#include "LofarUdpHeader.h"

/**
 * @file LofarUdpEmulator.h
 */

namespace pelican {

class ConfigNode;

namespace lofar {

/**
 * @class LofarUdpEmulator
 *  
 * @brief
 * 
 * @details
 * 
 */
class LofarUdpEmulator : public AbstractUdpEmulator
{
    public:
        /// Enum used to specify the sample type to use
        enum SampleType { i4complex = 4, i8complex = 8, i16complex = 16 };

        /// Constructor.
        LofarUdpEmulator(const ConfigNode& configNode);

        /// Destructor.
        ~LofarUdpEmulator() {}

        /// Fills the packet with data.
        void fillPacket();

        /// Returns a UDP packet.
        virtual void getPacketData(char*& ptr, unsigned long& size);

        /// Returns the interval between packets in microseconds.
        virtual unsigned long interval() {return _interval;}

        /// Returns the number of packets to send. Runs forever if negative.
        virtual int nPackets() {return _nPackets;}

        /// Sets the packet header.
        void setPacketHeader(UDPPacket::Header* udpPacketHeader);

        /// Set the emulator to loose even packets (to test reliability of receiver)
        void looseEvenPackets(bool loose);

        /// Returns the start delay in seconds.
        virtual int startDelay() {return _startDelay;}

    private:
        // Data Params
        int _subbandsPerPacket;   ///< Number of beamlets present in packet
        int _samplesPerPacket;    ///< Number of blocks per packet
        int _nrPolarisations;

        // Test Params
        SampleType    _sampleType;
        int           _nPackets;
        int           _startDelay;
        bool          _looseEvenPackets;

        unsigned long _packetCounter;   ///< Packet counter.
        unsigned long _packetSize;      ///< Actual packet size in bytes.
        unsigned long _interval;        ///< The interval between packets in microseconds.
        unsigned int  _clock;           ///< Station clock speed
        UDPPacket _packet;              ///< The packet to send.

        unsigned int _timestamp;        ///< The timestamp of the preceeding packet
        unsigned int _blockid;          ///< The blockSequenceNumber of the preceeding packet
};

} // namespace lofar
} // namespace pelican

#endif // LOFARUDPEMULATOR_H 
