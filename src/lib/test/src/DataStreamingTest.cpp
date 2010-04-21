#include "DataStreamingTest.h"

#include "LofarStreamDataClient.h"
#include "LofarServerClient.h"

#include "pelican/core/AbstractDataClient.h"
#include "pelican/utility/Config.h"
#include "pelican/utility/ConfigNode.h"

#include <QString>

#include "pelican/utility/memCheck.h"

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( DataStreamingTest );

/**
 *@details DataStreamingTest
 */
DataStreamingTest::DataStreamingTest()
    : CppUnit::TestFixture()
{
    // NOTE: Make this configurable??
    subbandsPerPacket = 32;
    samplesPerPacket = 2;
    nrPolarisations = 2;
    port = 8080;
    numPackets = 1000;
    usecDelay = 100000;
    sprintf(hostname, "%s", "127.0.0.1");
}


/**
 *@details
 */
DataStreamingTest::~DataStreamingTest()
{
}


void DataStreamingTest::setUp()
{
    // Define common configurations
    adapterXML =
        "<adapters>"
        "   <adapter name=\"test\">"
        "       <param value=\"2.0\"/>"
        "   </adapter>"
        "</adapters>";

    // set up dataTypes object
    // TODO
}


void DataStreamingTest::tearDown()
{
}


/**
 * @details
 * Tests setting up the lofar data generator.
 */
void DataStreamingTest::test_setupGenerator()
{
    // Use case: Setup LOFAR data emulator
    // Expect: Not to throw.
    try {
        dataGenerator.setDataParameters(subbandsPerPacket, samplesPerPacket, nrPolarisations);
        dataGenerator.connectBind(hostname, port);
    }
    catch(char* str) {
        QString error = QString("Could not set up DataStreamingTest: %1").arg(str);
        CPPUNIT_FAIL(error.toStdString());
    }
}


/*
 * Run the data streaming tests with the direct streaming client
 */
void DataStreamingTest::test_streamingClient()
{
    Config config;
    config.setFromString(adapterXML);
    Config::TreeAddress address;
    address << Config::NodeId("adapters", "");
    address << Config::NodeId("adapter", "test");
    ConfigNode configNode = config.get(address);
    LofarStreamDataClient* client = new LofarStreamDataClient(configNode);
    client->setDataRequirements(dataTypes);
    _testLofarDataClient(client);
    delete client;
}


/*
 * Run the data streaming tests with the server client
 */
void DataStreamingTest::test_serverClient()
{
    // Set up a server to listen to the data
    // TODO
    // Configure the dataClient
    Config config;
    config.setFromString(adapterXML);
    Config::TreeAddress address;
    address << Config::NodeId("adapters", "");
    address << Config::NodeId("adapter", "test");
    ConfigNode configNode = config.get(address);
    LofarServerClient* client = new LofarServerClient(configNode);
    client->setDataRequirements(dataTypes);

    // run the tests
    _testLofarDataClient(client);
    delete client;
}

void DataStreamingTest::_testLofarDataClient(AbstractDataClient* client)
{
    // TODO : write tests to ensure data is being received OK
    // client->getData();
}

} // namespace lofar
} // namespace pelican
