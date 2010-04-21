#include "test/LofarChunkerTest.h"
#include "LofarChunker.h"

#include "pelican/server/DataManager.h"
#include "pelican/utility/memCheck.h"

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( LofarChunkerTest );
// class LofarChunkerTest
LofarChunkerTest::LofarChunkerTest()
    : CppUnit::TestFixture()
{
}

LofarChunkerTest::~LofarChunkerTest()
{
}

void LofarChunkerTest::setUp()
{
    QString serverXml =
    "<buffers>"
    "   <LofarData>"
    "       <buffer maxSize=\"20000\" maxChunkSize=\"20000\"/>"
    "   </LofarData>"
    "</buffers>"
    ""
    "<chunkers>"
    "   <LofarChunker>"
    "       <data type=\"LofarData\"/>"
    "       <connection host=\"127.0.0.1\" port=\"8080\"/>"
    "       <params samplesPerPacket=\"64\" nrPolarisation=\"2\" subbandsPerPacket=\"4\"/>"
    "   </LofarChunker>"
    "</chunkers>";

    config.setFromString("", serverXml);
}

void LofarChunkerTest::tearDown()
{
}

void LofarChunkerTest::test_method()
{
    try {
        std::cout << "---------------------------------" << std::endl;
        std::cout << "Starting LofarChunker test" << std::endl;

        Config::TreeAddress address;
        address << Config::NodeId("server", "");
        address << Config::NodeId("chunkers", "");
        address << Config::NodeId("LofarChunker", "");
        ConfigNode configNode = config.get(address);

        LofarChunker chunker(configNode);
        chunker._nPackets = 1;
        QIODevice* device = chunker.newDevice();
        chunker.setDevice(device);

        pelican::DataManager dataManager(&config);
        dataManager.getStreamBuffer("LofarData");
        chunker.setDataManager(&dataManager);

        std::cout << "Chunker ready to test" << std::endl;

        chunker.next(device);

        std::cout << "Finished LofarChunker test" << std::endl;
    }
    catch (QString e) {
        CPPUNIT_FAIL("Unexpected exception: " + e.toStdString());
    }
}

} // namespace lofar
} // namespace pelican
