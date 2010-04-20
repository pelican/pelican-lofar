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
#include <cstdio>

namespace pelican {
namespace lofar {

/**
* @details
*/
template <typename SAMPLE_TYPE>
LofarDataGenerator<SAMPLE_TYPE>::LofarDataGenerator()
        : _subbandsPerPacket(1), _samplesPerPacket(1), _nrPolarisations(2),
          _fileDesc(-1), _packetHeader(NULL)
{
}


/**
* @details
*/
template <typename SAMPLE_TYPE>
LofarDataGenerator<SAMPLE_TYPE>::~LofarDataGenerator()
{
    // Close socket if open
    if (_fileDesc >= 0) {
        close(_fileDesc);
    }
}


/**
* @details
* Bind to the socket
*
* @param hostname
* @param port
*/
template <typename SAMPLE_TYPE>
void LofarDataGenerator<SAMPLE_TYPE>::connectBind(const char *hostname,
        short port)
{
    int retval;
    struct addrinfo hints;

    // use getaddrinfo, because gethostbyname is not thread safe.
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_INET;       // IPv4
    hints.ai_flags = AI_NUMERICSERV; // use only numeric port numbers
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_protocol = IPPROTO_UDP;

    char portStr[16];
    sprintf(portStr, "%hd", port );

    // Get address info
    if ((retval = getaddrinfo(hostname, portStr, &hints, &_addrInfo)) != 0 ) {
        fprintf(stderr, "getaddrinfo failed\n");
        throw "GetAddrInfo failed";
    }

    // result is a linked list of resolved addresses, we only use the first.
    if ((_fileDesc = socket(_addrInfo -> ai_family, _addrInfo -> ai_socktype, _addrInfo -> ai_protocol)) < 0) {
        fprintf(stderr, "Error creating socket\n");
        throw "Error creating socket";
    }

    // Connect to server.
    while (connect(_fileDesc, _addrInfo->ai_addr, _addrInfo->ai_addrlen) < 0) {
        if (errno == ECONNREFUSED) {
            if (sleep(1) > 0)
                // interrupted by a signal handler -- abort to allow this thread to
                // be forced to continue after receiving a SIGINT, as with any other
                // system call in this constructor
                fprintf(stderr, "Interrupted during sleep\n");
        }
        else {
            fprintf(stderr, "Error while trying to connect to server\n");
            throw "Error creating socket";
        }
    }
}


/**
* @details
* Set the UDP Packet header that will be used
*
* @param udpPacketHeader
*/
template <typename SAMPLE_TYPE>
void LofarDataGenerator<SAMPLE_TYPE>::setUdpPacketHeader(UDPPacket::Header* udpPacketHeader)
{
    _packetHeader = udpPacketHeader;
}


/**
* @details
* Set data parameters
*
* @param subbands
* @param samples
* @param polarisations
*/
template <typename SAMPLE_TYPE>
void LofarDataGenerator<SAMPLE_TYPE>::setDataParameters(int subbands,
        int samples, int polarisations)
{
    _subbandsPerPacket = subbands;
    _samplesPerPacket = samples;
    _nrPolarisations = polarisations;
}



/**
* @Details
* Send one packet
*/
template <typename SAMPLE_TYPE>
void LofarDataGenerator<SAMPLE_TYPE>::sendPacket()
{
    int packetSize = sizeof(struct UDPPacket::Header) + _subbandsPerPacket *
            _samplesPerPacket * _nrPolarisations * sizeof(SAMPLE_TYPE);
    socklen_t fromLen = sizeof(struct sockaddr_in);

    // Create test packet.
    UDPPacket* packet = (UDPPacket *) malloc(packetSize);

    // Create packet header.
    if (_packetHeader == NULL) {
        packet -> header.version    = 1;
        packet -> header.nrBeamlets = 1;
        packet -> header.nrBlocks   = 1;
        packet -> header.station    = 1;
        packet -> header.sourceInfo = 1;
    }
    else {
        // Copy header to packet.
        memcpy(&(packet -> header), _packetHeader, sizeof(UDPPacket::Header));
    }

    // Send test packet.
    if (sendto(_fileDesc, reinterpret_cast<char *> (packet), packetSize, 0, _addrInfo -> ai_addr, fromLen) < 0) {
        perror("Error sending packet");
        throw "Error sending packet";
    }
}


/**
* Send multiple packets
*
* @param numPackets
* @param usec
*/
template <typename SAMPLE_TYPE>
void LofarDataGenerator<SAMPLE_TYPE>::sendPackets(int numPackets,
        unsigned long usec)
{
    int packetSize = sizeof(struct UDPPacket::Header) + _subbandsPerPacket *
            _samplesPerPacket * _nrPolarisations * sizeof(SAMPLE_TYPE);
    socklen_t fromLen = sizeof(struct sockaddr_in);

    int packetCounter = 1;

    unsigned i, j;

    // Create test packet.
    UDPPacket* packet = (UDPPacket *) malloc(packetSize);

    // Create packet header.
    if (_packetHeader == NULL) {
        packet -> header.version    = 1;
        packet -> header.nrBeamlets = 1;
        packet -> header.nrBlocks   = 1;
        packet -> header.station    = 1;
        packet -> header.sourceInfo = 1;
    }
    else {
        // Copy header to packet.
        memcpy(&(packet -> header), _packetHeader, sizeof(UDPPacket::Header));
    }

    // Create test data in packet.
    SAMPLE_TYPE *s = reinterpret_cast<SAMPLE_TYPE *>(packet -> data);
    for (i = 0; i < _samplesPerPacket; i++) {
        for (j = 0; j < _subbandsPerPacket; j++) {
            s[i * _subbandsPerPacket * _nrPolarisations +
              j * _nrPolarisations] = SAMPLE_TYPE(j,j);
            s[i * _subbandsPerPacket * _nrPolarisations +
              j * _nrPolarisations + 1] = SAMPLE_TYPE(j,j);
        }
    }

    while(packetCounter < numPackets) {
        // Update packet timestamp.
        packet -> header.timestamp = packetCounter;
        packet -> header.blockSequenceNumber = packetCounter;

        int retval;
        if( (retval = sendto(_fileDesc, reinterpret_cast<char *> (packet), packetSize, 0, _addrInfo -> ai_addr, fromLen) < 0)) {
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

} // namepsace lofar
} // namepsace pelican
