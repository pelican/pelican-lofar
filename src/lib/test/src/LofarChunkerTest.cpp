#include "test/LofarChunkerTest.h"
#include "LofarUdpEmulator.h"
#include "LofarChunker.h"
#include "LofarUdpHeader.h"
#include "LofarTypes.h"

#include "pelican/emulator/EmulatorDriver.h"
#include "pelican/server/DataManager.h"
#include "pelican/comms/Data.h"
#include "pelican/server/test/ChunkerTester.h"

#include "pelican/utility/memCheck.h"

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
    QString bufferConfig =
            "<buffers>"
            "   <LofarData>"
            "       <buffer maxSize=\"100000000\" maxChunkSize=\"100000000\"/>"
            "   </LofarData>"
            "</buffers>";

    // Chunker configuration.
    _samplesPerPacket  = 32;   // Number of block per frame (for a 32 MHz beam).
    _nrPolarisations   = 2;    // Number of polarisation in the data.
    _numPackets        = 1000; // Number of packet to send.
    _clock             = 200;  // Rounded up clock station clock speed.
    _subbandsPerPacket = _clock == 200 ? 42 : 54;  // Number of blocks per frame.

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

    // Create the server configuration from the buffer and chunker.
    QString serverXml = bufferConfig + chunkerConfig;
    _config.setFromString("", serverXml);

    // Set up LOFAR data emulator configuration.
    unsigned interval = 1000;
    unsigned startDelay = 1;
    unsigned sampleBytes = 8;
    QString emulatorConfig = QString(
            "<LofarUdpEmulator>"
            "    <connection host=\"127.0.0.1\" port=\"8090\"/>"
            "    <params clock=\"%1\"/>"
            "    <packet interval=\"%2\""
            "            startDelay=\"%3\""
            "            sampleSize=\"%4\""
            "            samples=\"%5\""
            "            polarisations=\"%6\""
            "            subbands=\"%7\""
            "            nPackets=\"%8\"/>"
            "</LofarUdpEmulator>"
    )
    .arg(_clock)
    .arg(interval)
    .arg(startDelay)
    .arg(sampleBytes)
    .arg(_samplesPerPacket)
    .arg(_nrPolarisations)
    .arg(_subbandsPerPacket)
    .arg(_numPackets + 10);

    _emulatorNode.setFromString(emulatorConfig);
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
    typedef TYPES::i8complex i8c;
    double err = 1.0e-6;
    try {
        std::cout << "---------------------------------" << std::endl;
        std::cout << "Starting LofarChunker normalPackets test" << std::endl;

        // Get chunker configuration.
        Config::TreeAddress address;
        address << Config::NodeId("server", "");
        address << Config::NodeId("chunkers", "");
        address << Config::NodeId("LofarChunker", "");
        ConfigNode configNode = _config.get(address);

        // Create and setup chunker.
        LofarChunker chunker(configNode);
        QIODevice* device = chunker.newDevice();
        chunker.setDevice(device);

        // Create Data Manager.
        pelican::DataManager dataManager(&_config);
        dataManager.getStreamBuffer("LofarData");
        chunker.setDataManager(&dataManager);

        // Start Lofar Data Generator.
        EmulatorDriver emulator(new LofarUdpEmulator(_emulatorNode));

        // Acquire data through chunker.
        chunker.next(device);

        // Test read data.
        LockedData d = dataManager.getNext("LofarData");
        CPPUNIT_ASSERT(d.isValid());

        // Check the data in the chunk.
        char* data = (char *)reinterpret_cast<AbstractLockableData*>
                                            (d.object())->data()->data();

        UDPPacket *packet;
        unsigned packetSize = sizeof(struct UDPPacket::Header)
                + _subbandsPerPacket * _samplesPerPacket * _nrPolarisations
                * sizeof(i8c);

        unsigned idx = 0;
        for (int p = 0; p < _numPackets; ++p)
        {
            packet = (UDPPacket *) (data + packetSize * p);
            i8c* s = reinterpret_cast<i8c*>(&packet->data);

            for (int sb = 0; sb < _subbandsPerPacket; ++sb)
            {
                for (int t = 0; t < _samplesPerPacket; ++t)
                {
                    idx = _nrPolarisations * (t + sb * _samplesPerPacket);
                    // pol 1
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(float(sb), (float)s[idx].real(), err);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(float(0), (float)s[idx].imag(), err);

                    // pol 2
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(float(t), (float)s[idx + 1].real(), err);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL(float(1), (float)s[idx + 1].imag(), err);
                }
            }
        }

        std::cout << "Finished LofarChunker normalPackets test" << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }
    catch (const QString& e) {
        CPPUNIT_FAIL("Unexpected exception: " + e.toStdString());
    }
}

/**
* @details
* Test to check that lost packets are handled appropriately.
*/
void LofarChunkerTest::test_lostPackets()
{
    typedef TYPES::i8complex i8c;
    double err = 1.0e-6;
    try {
        std::cout << "---------------------------------" << std::endl;
        std::cout << "Starting LofarChunker lostPackets test" << std::endl;

        // Get chunker configuration.
        Config::TreeAddress address;
        address << Config::NodeId("server", "");
        address << Config::NodeId("chunkers", "");
        address << Config::NodeId("LofarChunker", "");
        ConfigNode configNode = _config.get(address);

        // Create and setup chunker.
        LofarChunker chunker(configNode);
        QIODevice* device = chunker.newDevice();
        chunker.setDevice(device);

        // Create Data Manager.
        pelican::DataManager dataManager(&_config);
        dataManager.getStreamBuffer("LofarData");
        chunker.setDataManager(&dataManager);

        // Start Lofar Data Generator.
        LofarUdpEmulator* emu = new LofarUdpEmulator(_emulatorNode);
        emu->looseEvenPackets(true);
        EmulatorDriver emulator(emu);

        // Acquire data through chunker.
        chunker.next(device);

        // Test read data
        LockedData d = dataManager.getNext("LofarData");

        CPPUNIT_ASSERT(d.isValid());

        char* data = (char *)(reinterpret_cast<AbstractLockableData*>
                                            (d.object())->data()->data());

        UDPPacket *packet;
        unsigned packetSize = sizeof(struct UDPPacket::Header)
                + _subbandsPerPacket * _samplesPerPacket * _nrPolarisations
                * sizeof(i8c);

        unsigned idx = 0;
        for (int p = 0; p < _numPackets; ++p)
        {
            packet = (UDPPacket *) (data + packetSize * p);
            i8c* s = reinterpret_cast<i8c*>(&packet->data);

            for (int sb = 0; sb < _subbandsPerPacket; ++sb)
            {
                for (int t = 0; t < _samplesPerPacket; ++t)
                {
                    idx = _nrPolarisations * (t + sb * _samplesPerPacket);
                    if (p % 2 == 1)
                    {
                        // pol 1
                        CPPUNIT_ASSERT_DOUBLES_EQUAL(float(sb), (float)s[idx].real(), err);
                        CPPUNIT_ASSERT_DOUBLES_EQUAL(float(0), (float)s[idx].imag(), err);
                        // pol 2
                        CPPUNIT_ASSERT_DOUBLES_EQUAL(float(t), (float)s[idx + 1].real(), err);
                        CPPUNIT_ASSERT_DOUBLES_EQUAL(float(1), (float)s[idx + 1].imag(), err);
                    }
                    else
                    {
                        // pol 1
                        CPPUNIT_ASSERT_DOUBLES_EQUAL(float(0), (float)s[idx].real(), err);
                        CPPUNIT_ASSERT_DOUBLES_EQUAL(float(0), (float)s[idx].imag(), err);
                        // pol 2
                        CPPUNIT_ASSERT_DOUBLES_EQUAL(float(0), (float)s[idx + 1].real(), err);
                        CPPUNIT_ASSERT_DOUBLES_EQUAL(float(0), (float)s[idx + 1].imag(), err);
                    }
                }
            }
        }

        std::cout << "Finished LofarChunker lostPackets test" << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }
    catch (const QString& e) {
        CPPUNIT_FAIL("Unexpected exception: " + e.toStdString());
    }
}


} // namespace lofar
} // namespace pelican
