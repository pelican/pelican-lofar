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
LofarChunkerTest::LofarChunkerTest() : CppUnit::TestFixture()
{
    _samplesPerPacket  = 32;   // Number of block per frame (for a 32 MHz beam)
    _nrPolarisations   = 2;    // Number of polarization in the data
    _numPackets        = 1000; // Number of packet to send
    _clock             = 200;  // Rounded up clock station clock speed
    _subbandsPerPacket = _clock == 200 ? 42 : 54;  //  Number of block per frame

    QString bufferConfig =
            "<buffers>"
            "   <LofarData>"
            "       <buffer maxSize=\"100000000\" maxChunkSize=\"100000000\"/>"
            "   </LofarData>"
            "</buffers>";

    QString chunkerConfig = QString(
            "<chunkers>"
            "   <LofarChunker>"
            "       <connection host=\"127.0.0.1\" port=\"8090\"/>"
            "       <data type=\"LofarData\"/>"
            ""
            "       <dataBitSize            value=\"%1\" />"
            "       <samplesPerPacket       value=\"%2\" />"
            "       <subbandsPerPacket      value=\"%3\" />"
            "       <nRawPolarisations      value=\"%4\" />"
            "       <clock                  value=\"%5\" />"
            "       <udpPacketsPerIteration value=\"%6\" />"
            ""
            "   </LofarChunker>"
            "</chunkers>")
            .arg(8)
            .arg(_samplesPerPacket)
            .arg(_subbandsPerPacket)
            .arg(_nrPolarisations)
            .arg(_clock)
            .arg(_numPackets);


    QString serverXml = bufferConfig + chunkerConfig;
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

        for (int counter = 0; counter < _numPackets; counter++) {

            packet = (UDPPacket *) (dataPtr + packetSize * counter);
            TYPES::i8complex* s = reinterpret_cast<TYPES::i8complex*>(&(packet->data));

            for (int k = 0; k < _samplesPerPacket; k++)
                for (int j = 0; j < _subbandsPerPacket; j++) {
                    float val = s[k * _subbandsPerPacket * _nrPolarisations +  j * _nrPolarisations].real();
                    CPPUNIT_ASSERT_EQUAL(float(k + j), val);
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
            TYPES::i8complex* s = reinterpret_cast<TYPES::i8complex*>(&packet->data);
            for (int k = 0; k < _samplesPerPacket; k++)
                for (int j = 0; j < _subbandsPerPacket; j++) {
                    unsigned index = k * _subbandsPerPacket * _nrPolarisations +
                            j * _nrPolarisations;
                    float val = s[index].real();
                    if (counter % 2 == 1)
                        CPPUNIT_ASSERT_EQUAL(float(k + j), val);
                    else
                        CPPUNIT_ASSERT_EQUAL(0.0f, val);
                }
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
