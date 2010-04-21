#include "LofarDataGenerator.h"

#include "LofarUdpHeader.h"
#include "LofarTypes.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <arpa/inet.h>
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
LofarDataGenerator::LofarDataGenerator()
        : _subbandsPerPacket(1), _samplesPerPacket(1), _nrPolarisations(2),
          _fileDesc(-1), _packetHeader(NULL)
{ }

/**
* @details
*/
LofarDataGenerator::~LofarDataGenerator()
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
void LofarDataGenerator::connectBind(const char *hostname,
        short port)
{
    int retval;
    struct addrinfo *addrInfo = (struct addrinfo *) malloc(sizeof(struct addrinfo)); 
    struct addrinfo *hints = (struct addrinfo *) malloc(sizeof(struct addrinfo));

    // use getaddrinfo, because gethostbyname is not thread safe.
    memset(hints, 0, sizeof(struct addrinfo));
    hints -> ai_family = AF_INET;       // IPv4
    hints -> ai_flags = AI_NUMERICSERV; // use only numeric port numbers
    hints -> ai_socktype = SOCK_DGRAM;
    hints -> ai_protocol = IPPROTO_UDP;

    char portStr[16];
    sprintf(portStr, "%hd", port + 1000 );

    // Get address info
    if ((retval = getaddrinfo(hostname, portStr, hints, &addrInfo)) != 0 ) {
        fprintf(stderr, "getaddrinfo failed\n");
        exit(-1);
    }

    // result is a linked list of resolved addresses, we only use the first
    if ((_fileDesc = socket(addrInfo -> ai_family, addrInfo -> ai_socktype, addrInfo -> ai_protocol)) < 0) {
        fprintf(stderr, "Error creating socket\n");
        exit(-1);
    }

    // Bind socket
    if (bind(_fileDesc, (struct sockaddr *) addrInfo -> ai_addr, sizeof(struct sockaddr)) == -1) {
        fprintf(stderr, "Error binding socket\n");
        exit(-1);
    }

    // Create receiver object
    memset((char *) &_receiver, 0, sizeof(struct sockaddr_in));
    _receiver.sin_port = htons(8090);
    if (inet_aton("127.0.0.1", &_receiver.sin_addr) == 0) {
        fprintf(stderr, "inet_aton() failed\n");
    }

}

/**
* @details
* Set the UDP Packet header that will be used
*
* @param udpPacketHeader
*/
void LofarDataGenerator::setUdpPacketHeader(UDPPacket::Header* udpPacketHeader)
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
void LofarDataGenerator::setDataParameters(int subbands,
        int samples, int polarisations)
{
    _subbandsPerPacket = subbands;
    _samplesPerPacket = samples;
    _nrPolarisations = polarisations;
}

/**
* @details
* Set test parameters
*
* @param numPackets
* @param usec
* @param startDelay
* @param sampleType
*/
void LofarDataGenerator::setTestParams(int numPackets,
        unsigned long usec, unsigned long startDelay, SampleType sampleType)
{
     _numPackets = numPackets;
     _usec       = usec;
     _startDelay = startDelay;
     _sampleType = sampleType;
}

/**
* Send LOFAR-style UDP packets
*
* @param numPackets
* @param usec
*/
void LofarDataGenerator::sendPackets(int numPackets,
        unsigned long usec, unsigned long startDelay, SampleType sampleType)
{
    int packetSize = sizeof(struct UDPPacket::Header) + _subbandsPerPacket *
            _samplesPerPacket * _nrPolarisations;
    socklen_t fromLen = sizeof(struct sockaddr_in);

    switch (sampleType) {
        case i4complex: packetSize += sizeof(TYPES::i4complex); break;
        case i8complex: packetSize += sizeof(TYPES::i8complex); break;
        case i16complex: packetSize += sizeof(TYPES::i16complex); break;
    }

    int packetCounter = 0;

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
    switch (sampleType) {
        case i4complex:  {
            TYPES::i4complex *s = reinterpret_cast<TYPES::i4complex *>(packet -> data);
            for (i = 0; i < _samplesPerPacket; i++) {
                for (j = 0; j < _subbandsPerPacket; j++) {
                   s[i * _subbandsPerPacket * _nrPolarisations +
                     j * _nrPolarisations] = TYPES::i4complex(j,j);
                   s[i * _subbandsPerPacket * _nrPolarisations +
                     j * _nrPolarisations + 1] = TYPES::i4complex(j,j);
                 }
             }
         }          
         case i8complex:  {
             TYPES::i8complex *s = reinterpret_cast<TYPES::i8complex *>(packet -> data);
             for (i = 0; i < _samplesPerPacket; i++) {
                 for (j = 0; j < _subbandsPerPacket; j++) {
                     s[i * _subbandsPerPacket * _nrPolarisations +
                       j * _nrPolarisations] = TYPES::i8complex(j,j);
                     s[i * _subbandsPerPacket * _nrPolarisations +
                       j * _nrPolarisations + 1] = TYPES::i8complex(j,j);
                  }
              }
         } 
         case i16complex:  {
             TYPES::i16complex *s = reinterpret_cast<TYPES::i16complex *>(packet -> data);
             for (i = 0; i < _samplesPerPacket; i++) {
                 for (j = 0; j < _subbandsPerPacket; j++) {
                     s[i * _subbandsPerPacket * _nrPolarisations +
                         j * _nrPolarisations] = TYPES::i16complex(j,j);
                     s[i * _subbandsPerPacket * _nrPolarisations +
                         j * _nrPolarisations + 1] = TYPES::i16complex(j,j);
                  }
              }
         } 
    }

    // Start delay before sending any packets
    sleep(startDelay);

    while(packetCounter < numPackets) {
        // Update packet timestamp.
        packet -> header.timestamp = packetCounter;
        packet -> header.blockSequenceNumber = packetCounter;

        int retval;
        if( (retval = sendto(_fileDesc, reinterpret_cast<char *> (packet), packetSize, 0,(struct sockaddr *) &_receiver, fromLen) < 0)) {
            perror("Error sending packet");
            throw "Error sending packet";
        }

        ++packetCounter;
        usleep(usec);
    }
}

void LofarDataGenerator::run()
{
    sendPackets(_numPackets, _usec, _startDelay, _sampleType);
}

} // namepsace lofar
} // namepsace pelican
