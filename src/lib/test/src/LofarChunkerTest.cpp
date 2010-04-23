#include "test/LofarChunkerTest.h"
#include "LofarDataGenerator.h"
#include "LofarChunker.h"
#include "LofarUdpHeader.h"

#include "pelican/server/DataManager.h"
#include "pelican/utility/memCheck.h"
#include "pelican/comms/Data.h"

#include <boost/shared_ptr.hpp>

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
                        // printf("%d, %d, %d\n", counter, k + j, s[k * _subbandsPerPacket * _nrPolarisations +
                        //   j * _nrPolarisations].real());
                        ;
        }

        std::cout << "Finished LofarChunker test" << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }
    catch (QString e) {
        CPPUNIT_FAIL("Unexpected exception: " + e.toStdString());
    }
}

} // namespace lofar
} // namespace pelican
