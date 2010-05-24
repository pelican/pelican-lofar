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
#include <iostream>

namespace pelican {
namespace lofar {

/**
* @details
*/
LofarDataGenerator::LofarDataGenerator()
        : QThread(), _subbandsPerPacket(1), _samplesPerPacket(1), _nrPolarisations(2),
          _fileDesc(-1), _packetHeader(NULL), _seqNumbers(NULL)
{ }

/**
* @details
*/
LofarDataGenerator::~LofarDataGenerator()
{
    // Wait for the thread to finish.
    releaseConnection();
    if (isRunning()) wait();
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
    if ((retval = getaddrinfo(hostname, portStr, hints, &addrInfo)) != 0 )
        throw "getaddrinfo failed\n";

    // result is a linked list of resolved addresses, we only use the first
    if ((_fileDesc = socket(addrInfo -> ai_family, addrInfo -> ai_socktype, addrInfo -> ai_protocol)) < 0)
        throw "Error creating socket\n";

    // Bind socket
    if (bind(_fileDesc, (struct sockaddr *) addrInfo -> ai_addr, sizeof(struct sockaddr)) == -1)
        throw "Error binding socket\n";

    // Create receiver object
    memset((char *) &_receiver, 0, sizeof(struct sockaddr_in));
    _receiver.sin_port = htons(8090);
    if (inet_aton("127.0.0.1", &_receiver.sin_addr) == 0) {
        fprintf(stderr, "inet_aton() failed\n");
    }

}

/**
* @details
* Release generator socket
*/
void LofarDataGenerator::releaseConnection()
{
    // Close socket if open
    if (_fileDesc >= 0) {
        close(_fileDesc);
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
void LofarDataGenerator::setTestParams(int numPackets, unsigned long usec, 
    unsigned long startDelay, SampleType sampleType, unsigned int *seqNumbers)
{
     _numPackets = numPackets;
     _usec       = usec;
     _startDelay = startDelay;
     _sampleType = sampleType;
     _seqNumbers = seqNumbers;
}

/**
* Send LOFAR-style UDP packets
*/
void LofarDataGenerator::run()
{
    int packetSize = sizeof(struct UDPPacket::Header) + _subbandsPerPacket *
            _samplesPerPacket * _nrPolarisations;

    socklen_t fromLen = sizeof(struct sockaddr_in);

    switch (_sampleType) {
        case i4complex: packetSize *= sizeof(TYPES::i4complex); break;
        case i8complex: packetSize *= sizeof(TYPES::i8complex); break;
        case i16complex: packetSize *= sizeof(TYPES::i16complex); break;
    }

    int packetCounter = 0;
    unsigned i, j;

    // Create test packet.
    UDPPacket* packet = (UDPPacket *) malloc(packetSize);
    std::cout << "LofarDataGenerator packet size: " <<  packetSize << std::endl;

    // Create packet header.
    if (_packetHeader == NULL) {
        packet -> header.version    = 1;
        packet -> header.nrBeamlets = 1;
        packet -> header.nrBlocks   = 1;
        packet -> header.station    = 1;
        packet -> header.sourceInfo = 1;
    }
    else
        // Copy header to packet.
        memcpy(&(packet -> header), _packetHeader, sizeof(UDPPacket::Header));


    // Create test data in packet.
    switch (_sampleType) {
        case i4complex:  {
            TYPES::i4complex *s = reinterpret_cast<TYPES::i4complex *>(packet -> data);
            for (i = 0; i < _samplesPerPacket; i++) {

                for (j = 0; j < _subbandsPerPacket; j++) {
                   s[i * _subbandsPerPacket * _nrPolarisations +
                     j * _nrPolarisations] = TYPES::i4complex(i + j, i);
                   s[i * _subbandsPerPacket * _nrPolarisations +
                     j * _nrPolarisations + 1] = TYPES::i4complex(i + j, i);
                 }
             }
             break;
         }          
         case i8complex:  {
             TYPES::i8complex *s = reinterpret_cast<TYPES::i8complex *>(packet -> data);
             for (i = 0; i < _samplesPerPacket; i++) {
                 for (j = 0; j < _subbandsPerPacket; j++) {
                     s[i * _subbandsPerPacket * _nrPolarisations +
                       j * _nrPolarisations] = TYPES::i8complex(i + j, i);
                     s[i * _subbandsPerPacket * _nrPolarisations +
                       j * _nrPolarisations + 1] = TYPES::i8complex(i + j, i);
                  }
              }
              break;
         } 
         case i16complex:  {
             TYPES::i16complex *s = reinterpret_cast<TYPES::i16complex *>(packet -> data);
             for (i = 0; i < _samplesPerPacket; i++) {
                 for (j = 0; j < _subbandsPerPacket; j++) {
                     s[i * _subbandsPerPacket * _nrPolarisations +
                         j * _nrPolarisations] = TYPES::i16complex(i + j, i);
                     s[i * _subbandsPerPacket * _nrPolarisations +
                         j * _nrPolarisations + 1] = TYPES::i16complex(i + j, i);
                  }
              }
              break;
         } 
    }

    // Start delay before sending any packets
    sleep(_startDelay);

    while(packetCounter < _numPackets) {

        // Update packet timestamp.
        if (_seqNumbers == NULL) {
            packet -> header.timestamp = packetCounter;
            packet -> header.blockSequenceNumber = packetCounter;
        } else {
            packet -> header.timestamp = _seqNumbers[packetCounter];
            packet -> header.blockSequenceNumber = _seqNumbers[packetCounter];
        }

        int retval;
        if( (retval = sendto(_fileDesc, reinterpret_cast<char *> (packet), 
                             packetSize, 0,(struct sockaddr *) &_receiver, fromLen) < 0))
            throw "LofarDataGenerator: Error sending packet";

        ++packetCounter;
        usleep(_usec);
    }
}

} // namepsace lofar
} // namepsace pelican
