#include "test/LofarChunkerTest.h"
#include "LofarUdpEmulator.h"
#include "LofarChunker.h"
#include "LofarUdpHeader.h"
#include "LofarTypes.h"

#include "pelican/emulator/EmulatorDriver.h"
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
{
    _samplesPerPacket  = 32;   // Number of block per frame (for a 32 MHz beam)
    _nrPolarisations   = 2;    // Number of polarization in the data
    _numPackets        = 10000;   // Number of packet to send
    _clock             = 200;  // Rounded up clock station clock speed
    _subbandsPerPacket = _clock == 200 ? 42 : 54;  //  Number of block per frame 
    
    QString serverXml =
    "<buffers>"
    "   <LofarData>"
    "       <buffer maxSize=\"100000000\" maxChunkSize=\"100000000\"/>"
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
    "               nSamples=\""          + QString::number(_numPackets * _samplesPerPacket) + "\""
    "               clock=\""             + QString::number(_clock)             + "\"/>"
    "       <samples type=\"8\" />"
    "   </LofarChunker>"
    "</chunkers>";

    config.setFromString("", serverXml);

    // Set up LOFAR data emulator configuration.
    _emulatorNode.setFromString(""
            "<LofarUdpEmulator>"
            "    <connection host=\"127.0.0.1\" port=\"8090\"/>"
            "    <params clock=\""         + QString::number(_clock)             + "\"/>"
            "    <packet interval=\"1000\""
            "            startDelay=\"1\""
            "            sampleSize=\"8\""
            "            samples=\""       + QString::number(_samplesPerPacket)  + "\""
            "            polarisations=\"" + QString::number(_nrPolarisations)   + "\""
            "            subbands=\""      + QString::number(_subbandsPerPacket) + "\""
            "            nPackets=\""      + QString::number(_numPackets + 10)        + "\"/>"
            "</LofarUdpEmulator>");
}

/**
 * @details
 * Destructor
 */
LofarChunkerTest::~LofarChunkerTest()
{
}

/**
 * @details
 * Sets up environment/objects for test class
 */
void LofarChunkerTest::setUp()
{
}

/**
 * @details
 * Destroys objects and reset environment
 */
void LofarChunkerTest::tearDown()
{ 
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
        EmulatorDriver emulator(new LofarUdpEmulator(_emulatorNode));

        // Acquire data through chunker
        chunker.next(device);

        // Test read data
        LockedData d = dataManager.getNext("LofarData");
        char* dataPtr = (char *)(reinterpret_cast<AbstractLockableData*>(d.object()) -> data() -> data() );

        UDPPacket *packet;
        unsigned packetSize = sizeof(struct UDPPacket::Header) + _subbandsPerPacket *
                              _samplesPerPacket * _nrPolarisations * sizeof(TYPES::i8complex);

        unsigned int val;
        for (int counter = 0; counter < _numPackets; counter++) {

            packet = (UDPPacket *) (dataPtr + packetSize * counter);
            TYPES::i8complex *s = reinterpret_cast<TYPES::i8complex *>( &(packet -> data));

            for (int k = 0; k < _samplesPerPacket; k++)
                 for (int j = 0; j < _subbandsPerPacket; j++) {
                     val = s[k * _subbandsPerPacket * _nrPolarisations +  j * _nrPolarisations].real();
                     CPPUNIT_ASSERT(k + j == val);
                 }
        }

        std::cout << "Finished LofarChunker normalPackets test" << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }
    catch (QString e) {
        CPPUNIT_FAIL("Unexpected exception: " + e.toStdString());
    }
}

/**
 * @details
 * Test to check that lost packets are handled appropriately.
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
        LofarUdpEmulator* emu = new LofarUdpEmulator(_emulatorNode);
        emu -> looseEvenPackets(true);
        EmulatorDriver emulator(emu);

        // Acquire data through chunker
        chunker.next(device);

        // Test read data
        LockedData d = dataManager.getNext("LofarData");
        char* dataPtr = (char *)(reinterpret_cast<AbstractLockableData*>(d.object()) -> data() -> data() );

        UDPPacket *packet;
        unsigned packetSize = sizeof(struct UDPPacket::Header) + _subbandsPerPacket *
                              _samplesPerPacket * _nrPolarisations * sizeof(TYPES::i8complex);

        for (int counter = 0; counter < _numPackets; counter++) {

            packet = (UDPPacket *) (dataPtr + packetSize * counter);
            TYPES::i8complex *s = reinterpret_cast<TYPES::i8complex *>( &(packet -> data));
            for (int k = 0; k < _samplesPerPacket; k++)
                for (int j = 0; j < _subbandsPerPacket; j++)
                   if (counter % 2 == 1)
                       CPPUNIT_ASSERT(k + j ==  s[k * _subbandsPerPacket * _nrPolarisations +
                                                  j * _nrPolarisations].real());
                   else 
                       CPPUNIT_ASSERT(s[k * _subbandsPerPacket * _nrPolarisations +
                                        j * _nrPolarisations].real() == 0);
        }

        std::cout << "Finished LofarChunker lostPackets test" << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }
    catch (QString e) {
        CPPUNIT_FAIL("Unexpected exception: " + e.toStdString());
     }
}

} // namespace lofar
} // namespace pelican
