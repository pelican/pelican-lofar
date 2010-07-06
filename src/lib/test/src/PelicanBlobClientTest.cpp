#include "PelicanBlobClientTest.h"
#include "PelicanBlobClient.h"
#include "ChannelisedStreamData.h"
#include "pelican/utility/ConfigNode.h"
#include "pelican/output/PelicanTCPBlobServer.h"

#include <QtCore/QCoreApplication>

namespace pelican {

namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( PelicanBlobClientTest );
/**
 *@details PelicanBlobClientTest 
 */
PelicanBlobClientTest::PelicanBlobClientTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
PelicanBlobClientTest::~PelicanBlobClientTest()
{
}

void PelicanBlobClientTest::setUp()
{
    int argc = 1;
    char *argv[] = {(char*)"pelican"};
    _app = new QCoreApplication(argc,argv);
}

void PelicanBlobClientTest::tearDown()
{
    delete _app;
}

void PelicanBlobClientTest::test_method()
{
    // Create and configure TCP server
    QString xml = "<PelicanTCPBlobServer>"
                  "   <connection port=\"0\"/>"  // 0 = find unused system port
                  "</PelicanTCPBlobServer>";
    ConfigNode config(xml);
    PelicanTCPBlobServer server(config);
    sleep(1);

    // Create a blob and fill it with data.
    ChannelisedStreamData blobToSend;
    blobToSend.resize(2, 2, 2);
    std::complex<double>* ptr = blobToSend.data();
    for (unsigned i = 0; i < blobToSend.size(); ++i) {
        ptr[i] = std::complex<double>(i, i);
    }

    // Create the pelican blob client.
    PelicanBlobClient client("ChannelisedStreamData", "127.0.0.1", server.serverPort());

    // ensure we are registered before continuing
    while( server.clientsForType("ChannelisedStreamData") == 0 )
    {
        sleep(1);
    }

    for (int i = 0; i < 10; ++i) {
        // Send the blob.
        server.send("ChannelisedStreamData", &blobToSend);

        // Receive into a data blob.
        ChannelisedStreamData blob;
        QHash<QString, DataBlob*> dataHash;
        dataHash.insert("ChannelisedStreamData", &blob);
        client.getData(dataHash);

        // Check the data.
        std::complex<double>* ptrRecv = blob.data();
        CPPUNIT_ASSERT(blob.size() == blobToSend.size());
        CPPUNIT_ASSERT(blob.size() > 0);
        for (unsigned i = 0; i < blob.size(); ++i) {
            CPPUNIT_ASSERT(ptrRecv[i] == std::complex<double>(i, i));
        }
    }
}

} // namespace lofar
} // namespace pelican
