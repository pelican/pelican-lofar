#ifndef LOFARDATAGENERATOR_H
#define LOFARDATAGENERATOR_H

#include "LofarUdpHeader.h"
#include "LofarTypes.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

#include <QThread>
#include <QObject>

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
 */

/// Enum used to specify the sample type to use
enum SampleType { i4complex = 4, i8complex = 8, i16complex = 16 };

class LofarDataGenerator: public QThread
{

    Q_OBJECT

    public:
        /// Constructs the lofar data generator.
        LofarDataGenerator();
        /// Destructor.
        ~LofarDataGenerator();

    public:
        /// Bind to the socket.
        void connectBind(const char* hostname, short port);
        /// Release socket connection
        void releaseConnection();
        /// Set the UDP Packet header that will be used.
        void setUdpPacketHeader(UDPPacket::Header* packetHeader);
        /// Set data parameter.
        void setDataParameters(int subbands, int samples, int polarisations);
        /// Set parameters for next test
        void setTestParams(int numPackets, unsigned long usec, unsigned long startDelay, 
                           SampleType sampleType, unsigned int *seqNumbers);
        // Starts the thread, sending packets
        virtual void run();
    protected:
        /// Threaded method


    private:
        struct sockaddr_in _receiver;
        UDPPacket::Header* _packetHeader;
        int _fileDesc;

        // Data Params
        int _subbandsPerPacket;
        int _samplesPerPacket;
        int _nrPolarisations;
        
        // Test Params
        SampleType    _sampleType;
        int           _numPackets;
        unsigned long _usec;
        unsigned long _startDelay;
        unsigned int  *_seqNumbers;
};


} // namespace lofar
} // namespace pelican

#endif
