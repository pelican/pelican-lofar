#include "LofarDataGenerator.h"
#include "LofarUdpHeader.h"
#include "LofarTypes.h"
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>
#include <errno.h>
#include <netdb.h>
#include <string.h>
#include <stdio.h>

namespace pelicanLofar {

template <typename SAMPLE_TYPE> LofarDataGenerator<SAMPLE_TYPE>::LofarDataGenerator()
    : subbandsPerPacket(1), samplesPerPacket(1), nrPolarisations(2), fileDesc(-1), packetHeader(NULL)
{ }

template <typename SAMPLE_TYPE> LofarDataGenerator<SAMPLE_TYPE>::~LofarDataGenerator()
{
    // Close socket if open
    if (fileDesc >= 0) {
        close(fileDesc);
    }
}

// Set the UDP Packet header that will be used
template <typename SAMPLE_TYPE> void LofarDataGenerator<SAMPLE_TYPE>::setUdpPacketHedear(UDPPacket::Header *udpPacketHeader)
{
    packetHeader = udpPacketHeader;
}

// Set data parameters 
template <typename SAMPLE_TYPE> void LofarDataGenerator<SAMPLE_TYPE>::setDataParameters(int subbands, int samples, int polarisations)
{
    subbandsPerPacket = subbands;
    samplesPerPacket = samples;
    nrPolarisations = polarisations;
}

// Bind to the socket
template <typename SAMPLE_TYPE> void LofarDataGenerator<SAMPLE_TYPE>::connectBind(const char *hostname, short port)
{
    int retval;
    struct addrinfo hints;

    // use getaddrinfo, because gethostbyname is not thread safe
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_INET;       // IPv4
    hints.ai_flags = AI_NUMERICSERV; // use only numeric port numbers
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_protocol = IPPROTO_UDP;

    char portStr[16];
    sprintf(portStr, "%hd", port );

    // Get address info
    if ((retval = getaddrinfo(hostname, portStr, &hints, &addrInfo)) != 0 ) {
        fprintf(stderr, "getaddrinfo failed\n");
        throw "GetAddrInfo failed";
    }

    // result is a linked list of resolved addresses, we only use the first
    if ((fileDesc = socket(addrInfo -> ai_family, addrInfo -> ai_socktype, addrInfo -> ai_protocol)) < 0) {
        fprintf(stderr, "Error creating socket\n");
        throw "Error creating socket";
    }

    // Connect to server
    while (connect(fileDesc, addrInfo->ai_addr, addrInfo->ai_addrlen) < 0)
        if (errno == ECONNREFUSED) {
            if (sleep(1) > 0)
                // interrupted by a signal handler -- abort to allow this thread to
                // be forced to continue after receiving a SIGINT, as with any other
                // system call in this constructor
                fprintf(stderr, "Interrupted during sleep\n");
        } else {
            fprintf(stderr, "Error while trying to connect to server\n");
            throw "Error creating socket";
        } 
}

// Send one packet
template <typename SAMPLE_TYPE> void LofarDataGenerator<SAMPLE_TYPE>::sendPacket()
{
    int             packetSize   = sizeof(struct UDPPacket::Header) + subbandsPerPacket *
                                   samplesPerPacket * nrPolarisations * sizeof(SAMPLE_TYPE);
    socklen_t       fromLen      = sizeof(struct sockaddr_in);

    // Create test packet
    UDPPacket* packet = (UDPPacket *) malloc(packetSize);

    // Create packet header
    if (packetHeader == NULL) {
        packet -> header.version    = 1;
        packet -> header.nrBeamlets = 1;
        packet -> header.nrBlocks   = 1;
        packet -> header.station    = 1;
        packet -> header.sourceInfo = 1;
    } else
        // Copy header to packet
        memcpy(&(packet -> header), packetHeader, sizeof(UDPPacket::Header));

    // Send test packet
    if( sendto(fileDesc, reinterpret_cast<char *> (packet), packetSize, 0, addrInfo -> ai_addr, fromLen) < 0) {
        perror("Error sending packet");
        throw "Error sending packet";
    }
}

// Send multiple packets
template <typename SAMPLE_TYPE> void LofarDataGenerator<SAMPLE_TYPE>::sendPackets(int numPackets, unsigned long usec)
{
    int             packetSize   = sizeof(struct UDPPacket::Header) + subbandsPerPacket *
                                   samplesPerPacket * nrPolarisations * sizeof(SAMPLE_TYPE);
    socklen_t       fromLen      = sizeof(struct sockaddr_in);

    int packetCounter = 1;

    unsigned i, j;

    // Create test packet
    UDPPacket* packet = (UDPPacket *) malloc(packetSize);

    // Create packet header
    if (packetHeader == NULL) {
        packet -> header.version    = 1;
        packet -> header.nrBeamlets = 1;
        packet -> header.nrBlocks   = 1;
        packet -> header.station    = 1;
        packet -> header.sourceInfo = 1;
    } else
        // Copy header to packet
        memcpy(&(packet -> header), packetHeader, sizeof(UDPPacket::Header));

    // Create test data in packet
    SAMPLE_TYPE *s = reinterpret_cast<SAMPLE_TYPE *>(packet -> data);
    for(i = 0; i < samplesPerPacket; i++)
        for(j = 0; j < subbandsPerPacket; j++) {
            s[i * subbandsPerPacket * nrPolarisations +
              j * nrPolarisations] = SAMPLE_TYPE(j,j);
            s[i * subbandsPerPacket * nrPolarisations +
              j * nrPolarisations + 1] = SAMPLE_TYPE(j,j);
        }

    while(packetCounter < numPackets) {
        // Update packet timestamp
        packet -> header.timestamp = packetCounter;
        packet -> header.blockSequenceNumber = packetCounter;

        int retval;
        if( (retval = sendto(fileDesc, reinterpret_cast<char *> (packet), packetSize, 0, addrInfo -> ai_addr, fromLen) < 0)) {
            perror("Error sending packet");
            throw "Error sending packet";
        }

        ++packetCounter;
	usleep(usec);
    }
}

template class LofarDataGenerator<TYPES::i4complex>;
template class LofarDataGenerator<TYPES::i8complex>;
template class LofarDataGenerator<TYPES::i16complex>;

} // namepsace pelicanLofar
