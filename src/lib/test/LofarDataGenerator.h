#include "LofarUdpHeader.h"
#include "LofarTypes.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

/**
 * @file LofarDataGenerator.h
 */


namespace pelicanLofar {

/**
 * @class LofarDataGenerator
 *  
 * @brief
 *   class to generate LOFAR-type UDP packets, for testing purposes
 * @details
 * 
 */

template<typename SAMPLE_TYPE> class LofarDataGenerator {

    public:

	void    connectBind(const char *hostname, short port);
        void    setUdpPacketHedear(UDPPacket::Header *packetHeader);
        void    setDataParameters(int subbands, int samples, int polarisations);
        void    sendPacket();
        void    sendPackets(int numPackets, unsigned long usec);

    public:
	LofarDataGenerator();
	~LofarDataGenerator();

    private:
        struct addrinfo *       addrInfo;
        UDPPacket::Header *     packetHeader;
        int	                fileDesc, subbandsPerPacket, samplesPerPacket, nrPolarisations;

    // This is the static class function that serves as a C style function pointer
    // for the pthread_create call
    static void* start_thread(void *obj)
    {
        //Call run() function to do the ctual work
        reinterpret_cast<LofarDataGenerator<SAMPLE_TYPE> *>(obj) -> sendPackets(1,1);
        return NULL;
    }	        
};

} //namespace pelicanLofar
