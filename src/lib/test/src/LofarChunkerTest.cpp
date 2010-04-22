#include "test/LofarChunkerTest.h"
#include "LofarDataGenerator.h"
#include "LofarChunker.h"

#include "pelican/server/DataManager.h"
#include "pelican/utility/memCheck.h"

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( LofarChunkerTest );
// class LofarChunkerTest
LofarChunkerTest::LofarChunkerTest()
    : CppUnit::TestFixture()
{ }

LofarChunkerTest::~LofarChunkerTest()
{
}

void LofarChunkerTest::setUp()
{
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
    "       <params samplesPerPacket=\"64\" nrPolarisation=\"2\" subbandsPerPacket=\"4\" nPackets=\"1\"/>"
    "       <samples type=\"8\" />"
    "   </LofarChunker>"
    "</chunkers>";

    config.setFromString("", serverXml);

    // Setup LOFAR data emulator
    try {
        dataGenerator.setDataParameters(4, 64, 2);
        dataGenerator.connectBind("127.0.0.1", 8090);
    }
    catch(char* str) {
        QString error = QString("Could not set up LofarChunkerTest: %1").arg(str);
        CPPUNIT_FAIL(error.toStdString());
    }

}

void LofarChunkerTest::tearDown()
{
}

void LofarChunkerTest::test_method()
{
    try {
        
        std::cout << "---------------------------------" << std::endl;
        std::cout << "Starting LofarChunker test" << std::endl;

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
        dataGenerator.setTestParams(1, 100000, 1, i8complex);
        dataGenerator.start();

        // Acquire data through chunker
        chunker.next(device);

        // Wait for data generator to exit
        dataGenerator.wait(100);

        // TODO: Test read data!!

        std::cout << "Finished LofarChunker test" << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }
    catch (QString e) {
        CPPUNIT_FAIL("Unexpected exception: " + e.toStdString());
    }
}

} // namespace lofar
} // namespace pelican
