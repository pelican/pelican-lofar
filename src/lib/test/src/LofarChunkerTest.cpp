#include "test/LofarChunkerTest.h"
#include "LofarDataGenerator.h"
#include "LofarChunker.h"
#include "LofarUdpHeader.h"
#include "LofarTypes.h"

#include "pelican/server/DataManager.h"
#include "pelican/utility/memCheck.h"
#include "pelican/comms/Data.h"

#include <boost/shared_ptr.hpp>

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( LofarChunkerTest );

/**
 * @details
 * Constructor
 */
LofarChunkerTest::LofarChunkerTest()
    : CppUnit::TestFixture()
{ }

/**
 * @details
 * Desctructor
 */
LofarChunkerTest::~LofarChunkerTest()
{ }

/**
 * @details
 * Sets up environment/objects for test class
 */
void LofarChunkerTest::setUp()
{
    _subbandsPerPacket = 4;
    _samplesPerPacket  = 64;
    _nrPolarisations   = 2;
    _numPackets        = 10;
    
    QString serverXml =
    "<buffers>"
    "   <LofarData>"
    "       <buffer maxSize=\"20000000\" maxChunkSize=\"200000\"/>"
    "   </LofarData>"
    "</buffers>"
    ""
    "<chunkers>"
    "   <LofarChunker>"
    "       <data type=\"LofarData\"/>"
    "       <connection host=\"127.0.0.1\" port=\"8090\"/>"
    "       <params samplesPerPacket=\""  + QString::number(_samplesPerPacket)  + "\""
    "               nrPolarisation=\""    + QString::number(_nrPolarisations)   + "\""
    "               subbandsPerPacket=\"" + QString::number(_subbandsPerPacket) + "\""
    "               nPackets=\""          + QString::number(_numPackets)        + "\"/>"
    "       <samples type=\"8\" />"
    "   </LofarChunker>"
    "</chunkers>";

    config.setFromString("", serverXml);

    // Setup LOFAR data emulator
    try {
        dataGenerator.setDataParameters(_subbandsPerPacket, _samplesPerPacket, _nrPolarisations);
        dataGenerator.connectBind("127.0.0.1", 8090);
    }
    catch(char* str) {
        QString error = QString("Could not set up LofarChunkerTest: %1").arg(str);
        CPPUNIT_FAIL(error.toStdString());
    }

}

/**
 * @details
 * Destroys objects and reset environment
 */
void LofarChunkerTest::tearDown()
{ 
    // Cleanup LofarDataGenerator object
    dataGenerator.releaseConnection();
}

/**
 * @details
 * Test to check that normal packets are read correctly
 */
void LofarChunkerTest::test_normalPackets()
{
    try {
        
        std::cout << "---------------------------------" << std::endl;
        std::cout << "Starting LofarChunker normalPackets test" << std::endl;

        // Get chunker configuration
        Config::TreeAddress address;
        address << Config::NodeId("server", "");
        address << Config::NodeId("chunkers", "");
        address << Config::NodeId("LofarChunker", "");
        ConfigNode configNode = config.get(address);

        // Create and setup chunker
        LofarChunker chunker(configNode);
        QIODevice* device = chunker.newDevice();
        chunker.setDevice(device);

        // Create Data Manager
        pelican::DataManager dataManager(&config);
        dataManager.getStreamBuffer("LofarData");
        chunker.setDataManager(&dataManager);

        // Start Lofar Data Generator
        dataGenerator.setTestParams(_numPackets, 100000, 1, i8complex, NULL);
        dataGenerator.start();

        // Acquire data through chunker
        chunker.next(device);

        // Test read data
        LockedData d = dataManager.getNext("LofarData");
        char* dataPtr = (char *)(reinterpret_cast<AbstractLockableData*>(d.object()) -> data() -> data() );

        int counter, j, k;
        UDPPacket *packet;

        unsigned packetSize = sizeof(struct UDPPacket::Header) + _subbandsPerPacket *
                              _samplesPerPacket * _nrPolarisations * sizeof(TYPES::i8complex);

        for (counter = 0; counter < _numPackets; counter++) {

            packet = (UDPPacket *) (dataPtr + packetSize * counter);
            TYPES::i8complex *s = reinterpret_cast<TYPES::i8complex *>( &(packet -> data));

            for (k = 0; k < _samplesPerPacket; k++)
                 for (j = 0; j < _subbandsPerPacket; j++)
                     CPPUNIT_ASSERT(k + j ==  s[k * _subbandsPerPacket * _nrPolarisations +
                                                j * _nrPolarisations].real());
        }

        device -> close();

        std::cout << "Finished LofarChunker normalPackets test" << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }
    catch (QString e) {
        CPPUNIT_FAIL("Unexpected exception: " + e.toStdString());
    }
}

/**
 * @details
 * Test to check that lost packets are handled appropriately
 */
void LofarChunkerTest::test_lostPackets()
{
    try {
        
        std::cout << "---------------------------------" << std::endl;
        std::cout << "Starting LofarChunker lostPackets test" << std::endl;

        // Get chunker configuration
        Config::TreeAddress address;
        address << Config::NodeId("server", "");
        address << Config::NodeId("chunkers", "");
        address << Config::NodeId("LofarChunker", "");
        ConfigNode configNode = config.get(address);

        // Create and setup chunker
        LofarChunker chunker(configNode);
        QIODevice* device = chunker.newDevice();
        chunker.setDevice(device);

        // Create Data Manager
        pelican::DataManager dataManager(&config);
        dataManager.getStreamBuffer("LofarData");
        chunker.setDataManager(&dataManager);

        // Start Lofar Data Generator
        unsigned int seqs[10] = {0, 1, 4, 4, 4, 5, 6, 6, 8, 9 };
        dataGenerator.setTestParams(_numPackets, 100000, 1, i8complex, (unsigned int*) &seqs);
        dataGenerator.start();

        // Acquire data through chunker
        chunker.next(device);

        // Test read data
        LockedData d = dataManager.getNext("LofarData");
        char* dataPtr = (char *)(reinterpret_cast<AbstractLockableData*>(d.object()) -> data() -> data() );

        int counter, j, k;
        UDPPacket *packet;

        unsigned packetSize = sizeof(struct UDPPacket::Header) + _subbandsPerPacket *
                              _samplesPerPacket * _nrPolarisations * sizeof(TYPES::i8complex);

        // Packet seqid should be as follows:
        unsigned int storedSeqid[10] = { 0, 1, 0, 0, 4, 5, 6, 0, 8, 9 };

        for (counter = 0; counter < _numPackets; counter++) {

            packet = (UDPPacket *) (dataPtr + packetSize * counter);
            TYPES::i8complex *s = reinterpret_cast<TYPES::i8complex *>( &(packet -> data));

            CPPUNIT_ASSERT(storedSeqid[counter] == packet -> header.timestamp);

            for (k = 0; k < _samplesPerPacket; k++)
                for (j = 0; j < _subbandsPerPacket; j++)
                   if (storedSeqid[counter] != 0 || counter == 0)
                       CPPUNIT_ASSERT(k + j ==  s[k * _subbandsPerPacket * _nrPolarisations +
                                                  j * _nrPolarisations].real());
                   else
                       CPPUNIT_ASSERT(s[k * _subbandsPerPacket * _nrPolarisations +
                                        j * _nrPolarisations].real() == 0);
        }

        device -> close();

        std::cout << "Finished LofarChunker lostPackets test" << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }
    catch (QString e) {
        CPPUNIT_FAIL("Unexpected exception: " + e.toStdString());
     }
}

} // namespace lofar
} // namespace pelican
