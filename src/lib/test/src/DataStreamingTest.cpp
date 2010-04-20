#include "DataStreamingTest.h"
#include "LofarStreamDataClient.h"
#include "LofarServerClient.h"
#include "pelican/core/AbstractDataClient.h"
#include "pelican/utility/Config.h"
#include "pelican/utility/ConfigNode.h"
#include "pelican/utility/memCheck.h"

namespace pelicanLofar {

CPPUNIT_TEST_SUITE_REGISTRATION( DataStreamingTest );
/**
 *@details DataStreamingTest 
 */
DataStreamingTest::DataStreamingTest()
    : CppUnit::TestFixture()
{
    //NOTE: Make this configurable??
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
    // Setup LOFAR data emulator
    try {
        LofarDataGenerator<TYPES::i8complex> generator;
        generator.setDataParameters(subbandsPerPacket, samplesPerPacket, nrPolarisations);
        generator.connectBind(hostname, port);
        dataGenerator = (void *) &generator;
    } catch(char * str) {
        fprintf(stderr, "Could not set up DataStreamingTest: %s\n", str);
        throw str; 
    }

    // define common configurations
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
//    LofarStreamDataClient* client = new LofarStreamDataClient(configNode, dataTypes);
//    _testLofarDataClient(client);
//    delete client;
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
//    LofarServerClient* client = new LofarServerClient(configNode, dataTypes);

    // run the tests
//    _testLofarDataClient(client);
//    delete client;
}

void _testLofarDataClient(pelican::AbstractDataClient* client)
{
    // TODO : write tests to ensure data is being received OK
    // client->getData();
}

} // namespace pelicanLofar
