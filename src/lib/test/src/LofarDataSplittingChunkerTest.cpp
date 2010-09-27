#include "test/LofarDataSplittingChunkerTest.h"

#include "LofarDataSplittingChunker.h"
#include "LofarUdpEmulator.h"
#include "LofarUdpHeader.h"
#include "LofarTypes.h"

#include "pelican/emulator/EmulatorDriver.h"
#include "pelican/server/DataManager.h"
#include "pelican/comms/Data.h"
#include "pelican/server/test/ChunkerTester.h"

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION(LofarDataSplittingChunkerTest);

/**
* @details
* Constructor
*/
LofarDataSplittingChunkerTest::LofarDataSplittingChunkerTest()
: CppUnit::TestFixture()
{
    _clock = 200;
    _sampleBits = 8;

    _nSamples = 16;
    _nPols = 2;
    _nSubbands = 61;//(_clock == 200) ? 42 : 54;

    _nPackets = 1000;

    _host = "127.0.0.1";
    _port = 8090;
    _chunkType1 = "LofarChunkData1";
    _chunkType2 = "LofarChunkData2";

    // Setup the Chunker configuration.
    QString chunkerConfig =
            "<chunkers>"
            ""
            "   <LofarDataSplittingChunker>"
            ""
            "       <connection host=\"%1\" port=\"%2\"/>"
            "       <data type=\"%3\"/>"
            "       <data type=\"%4\"/>"
            ""
            "       <dataBitSize            value=\"%5\" />"
            "       <samplesPerPacket       value=\"%6\" />"
            "       <subbandsPerPacket      value=\"%7\" />"
            "       <nRawPolarisations      value=\"%8\" />"
            "       <clock                  value=\"%9\" />"
            "       <udpPacketsPerIteration value=\"%10\" />"
            ""
            "   </LofarDataSplittingChunker>"
            ""
            "</chunkers>";
    chunkerConfig = chunkerConfig.arg(_host);       // 1
    chunkerConfig = chunkerConfig.arg(_port);       // 2
    chunkerConfig = chunkerConfig.arg(_chunkType1); // 3
    chunkerConfig = chunkerConfig.arg(_chunkType2); // 4

    chunkerConfig = chunkerConfig.arg(_sampleBits); // 5
    chunkerConfig = chunkerConfig.arg(_nSamples);   // 6
    chunkerConfig = chunkerConfig.arg(_nSubbands);  // 7
    chunkerConfig = chunkerConfig.arg(_nPols);      // 8
    chunkerConfig = chunkerConfig.arg(_clock);      // 9
    chunkerConfig = chunkerConfig.arg(_nPackets);   // 10

    // Configuration of buffers the chunker is writing to.
    QString bufferConfig =
            "<buffers>"
            ""
            "   <%1>"
            "       <buffer maxSize=\"100000000\" maxChunkSize=\"100000000\"/>"
            "   </%1>"
            ""
            "   <%2>"
            "       <buffer maxSize=\"100000000\" maxChunkSize=\"100000000\"/>"
            "   </%2>"
            ""
            "</buffers>";
    bufferConfig = bufferConfig.arg(_chunkType1);
    bufferConfig = bufferConfig.arg(_chunkType2);

    // Create the server configuration from the chunker and buffer
    QString serverXml = chunkerConfig + bufferConfig;
    _config.setFromString("", serverXml);

    // Set up tje LOFAR data emulator configuration.
    unsigned interval = 1000; // Target packet interval in microseconds.
    unsigned startDelay = 1;
    QString emulatorConfig =
            "<LofarUdpEmulator>"
            "    <connection host=\"%1\" port=\"%2\"/>"
            "    <params clock=\"%3\"/>"
            "    <packet interval=\"%4\""
            "            startDelay=\"%5\""
            "            sampleSize=\"%6\""
            "            samples=\"%7\""
            "            polarisations=\"%8\""
            "            subbands=\"%9\""
            "            nPackets=\"%10\"/>"
            "</LofarUdpEmulator>";
    emulatorConfig = emulatorConfig.arg(_host);          // 1
    emulatorConfig = emulatorConfig.arg(_port);          // 2
    emulatorConfig = emulatorConfig.arg(_clock);         // 3
    emulatorConfig = emulatorConfig.arg(interval);       // 4
    emulatorConfig = emulatorConfig.arg(startDelay);     // 5
    emulatorConfig = emulatorConfig.arg(_sampleBits);    // 6
    emulatorConfig = emulatorConfig.arg(_nSamples);      // 7
    emulatorConfig = emulatorConfig.arg(_nPols);         // 8
    emulatorConfig = emulatorConfig.arg(_nSubbands);     // 9
    emulatorConfig = emulatorConfig.arg(_nPackets + 10); // 10

    _emulatorNode.setFromString(emulatorConfig);
}



/**
* @details
* Destructor
*/
LofarDataSplittingChunkerTest::~LofarDataSplittingChunkerTest()
{
}

/**
* @details
* Sets up environment/objects for test class
*/
void LofarDataSplittingChunkerTest::setUp()
{
}

/**
* @details
* Destroys objects and reset environment
*/
void LofarDataSplittingChunkerTest::tearDown()
{
}

/**
* @details
* Test to check that normal packets are read correctly
*/
void LofarDataSplittingChunkerTest::test_normal_packets()
{
    try {
        cout << endl;
        cout << "[START] LofarDataSplittingChunkerTest::test_normal_packets()";
        cout << endl;

        // Get chunker configuration.
        //_config.summary();
        Config::TreeAddress address;
        address << Config::NodeId("server", "");
        address << Config::NodeId("chunkers", "");
        address << Config::NodeId("LofarDataSplittingChunker", "");
        ConfigNode configNode = _config.get(address);

        // Create and setup chunker.
        LofarDataSplittingChunker chunker(configNode);
        QIODevice* device = chunker.newDevice();
        chunker.setDevice(device);

        // Create Data Manager.
        pelican::DataManager dataManager(&_config);
        dataManager.getStreamBuffer("LofarChunkData1");
        dataManager.getStreamBuffer("LofarChunkData2");
        chunker.setDataManager(&dataManager);

        // Start Lofar Data Generator.
        EmulatorDriver emulator(new LofarUdpEmulator(_emulatorNode));

        // Acquire data through chunker.
        chunker.next(device);


        // Read through the data in the chunks and check that it is correct.
        typedef TYPES::i8complex i8c;

        // Chunk 1.
        cout << "- Checking chunk 1." << endl;
        LockedData d = dataManager.getNext("LofarChunkData1");
        char* data = (char*)reinterpret_cast<AbstractLockableData*>
                                                (d.object())->data()->data();
        CPPUNIT_ASSERT(d.isValid());

        UDPPacket* packet;
        size_t packetSize = sizeof(struct UDPPacket::Header)
                + _nSubbands * _nSamples * _nPols * sizeof(i8c);

        for (unsigned p = 0; p < _nPackets; ++p)
        {
            packet = (UDPPacket*)(data + packetSize * p);
            i8c* s = reinterpret_cast<i8c*>(&packet->data);

            unsigned idx;
            for (unsigned i = 0; i < _nSamples; ++i)
            {
                for (unsigned j = 0; j < _nSubbands; ++j)
                {
                    idx = _nPols * (j + i * _nSubbands);
                    CPPUNIT_ASSERT_EQUAL(float(i + j), (float)s[idx].real());
                    CPPUNIT_ASSERT_EQUAL(float(i), (float)s[idx].imag());

                    CPPUNIT_ASSERT_EQUAL(float(i + j), (float)s[idx + 1].real());
                    CPPUNIT_ASSERT_EQUAL(float(j), (float)s[idx + 1].imag());
                }
            }
        }


        // Chunk 2.
        cout << "- Checking chunk 2 (TODO)." << endl;
        d = dataManager.getNext("LofarChunkData2");
        data = (char*)reinterpret_cast<AbstractLockableData*>
                                    (d.object())->data()->data();
        CPPUNIT_ASSERT(d.isValid());


        cout << "[DONE] LofarDataSplittingChunkerTest::test_normal_packets()";
        cout << endl;
    }

    catch (const QString& e)
    {
        CPPUNIT_FAIL("ERROR: " + e.toStdString());
    }
}

} // namespace lofar
} // namespace pelican
