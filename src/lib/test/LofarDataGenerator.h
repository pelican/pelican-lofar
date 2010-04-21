#include "LofarUdpHeader.h"
#include "LofarTypes.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

/**
 * @file LofarDataGenerator.h
 */

namespace pelican {
namespace lofar {

/**
 * @class LofarDataGenerator
 *
 * @brief
 * Class to generate LOFAR-type UDP packets, for testing purposes.
 *
 * @details
 *
 */

template<typename SAMPLE_TYPE> class LofarDataGenerator
{
    public:
        /// Constructs the lofar data generator.
        LofarDataGenerator();
        /// Destructor.
        ~LofarDataGenerator();

    public:
        /// Bind to the socket.
        void connectBind(const char* hostname, short port);
        /// Set the UDP Packet header that will be used.
        void setUdpPacketHeader(UDPPacket::Header* packetHeader);
        /// Set data parameter.
        void setDataParameters(int subbands, int samples, int polarisations);
        /// Send a data packet.
        void sendPacket();
        /// Send a number of data packets.
        void sendPackets(int numPackets, unsigned long usec);

    private:
        struct sockaddr_in _receiver;
        UDPPacket::Header* _packetHeader;
        int _fileDesc;
        int _subbandsPerPacket;
        int _samplesPerPacket;
        int _nrPolarisations;

        // This is the static class function that serves as a C style function
        // pointer for the pthread_create call
        static void* start_thread(void* obj)
        {
            // Call run() function to do the actual work
            reinterpret_cast<LofarDataGenerator<SAMPLE_TYPE> *>(obj)->sendPackets(1, 1);
            return NULL;
        }
};

} // namespace lofar
} // namespace pelican
