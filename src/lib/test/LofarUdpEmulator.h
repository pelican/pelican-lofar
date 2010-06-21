#ifndef LOFARUDPEMULATOR_H
#define LOFARUDPEMULATOR_H

#include "pelican/emulator/AbstractUdpEmulator.h"
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

        /// Sets the sequence numbers for the packet header.
        void setSeqNumbers(unsigned int *seqNumbers);

        /// Returns the start delay in seconds.
        virtual int startDelay() {return _startDelay;}

    private:
        // Data Params
        int _subbandsPerPacket;
        int _samplesPerPacket;
        int _nrPolarisations;

        // Test Params
        SampleType    _sampleType;
        int           _nPackets;
        int           _startDelay;
        unsigned int* _seqNumbers;

        unsigned long _packetCounter; ///< Packet counter.
        unsigned long _packetSize;    ///< Actual packet size in bytes.
        unsigned long _interval;      ///< The interval between packets in microseconds.
        UDPPacket _packet;            ///< The packet to send.
};

} // namespace lofar
} // namespace pelican

#endif // LOFARUDPEMULATOR_H 
