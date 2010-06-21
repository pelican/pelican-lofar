#include "DataStreamingTest.h"

#include "LofarStreamDataClient.h"
#include "LofarServerClient.h"
#include "LofarUdpEmulator.h"

#include "pelican/core/AbstractDataClient.h"
#include "pelican/emulator/EmulatorDriver.h"
#include "pelican/utility/Config.h"
#include "pelican/utility/ConfigNode.h"

#include <QtCore/QString>

#include "pelican/utility/memCheck.h"

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( DataStreamingTest );

/**
 * @details
 * DataStreamingTest constructor.
 */
DataStreamingTest::DataStreamingTest()
    : CppUnit::TestFixture()
{
    // NOTE: Make this configurable??
    _subbandsPerPacket = 32;
    _samplesPerPacket = 2;
    _nrPolarisations = 2;
    _port = 8080;
    _numPackets = 1000;
    _interval = 100000;
    _hostname = "127.0.0.1";

    // Set up LOFAR data emulator configuration.
    _emulatorNode.setFromString(""
            "<LofarUdpEmulator>"
            "    <connection host=\"" + _hostname + "\" port=\"" + QString::number(_port) + "\"/>"
            "    <packet interval=\""      + QString::number(_interval)  + "\""
            "            startDelay=\"1\""
            "            sampleSize=\"8\""
            "            samples=\""       + QString::number(_samplesPerPacket)  + "\""
            "            polarisations=\"" + QString::number(_nrPolarisations)   + "\""
            "            subbands=\""      + QString::number(_subbandsPerPacket) + "\""
            "            nPackets=\""      + QString::number(_numPackets)        + "\"/>"
            "</LofarUdpEmulator>");
}

/**
 * @details
 * Destroys the data streaming test suite.
 */
DataStreamingTest::~DataStreamingTest()
{
}

/**
 * @details
 * Set-up routine called before each test.
 */
void DataStreamingTest::setUp()
{
    // Define common configurations
    _adapterXML =
        "<adapters>"
        "   <adapter name=\"test\">"
        "       <param value=\"2.0\"/>"
        "   </adapter>"
        "</adapters>";

    // TODO Set up dataTypes object
}

/**
 * @details
 * Clean-up routine called after each test.
 */
void DataStreamingTest::tearDown()
{
}

/**
 * @details
 * Tests setting up the Lofar data generator.
 */
void DataStreamingTest::test_setupGenerator()
{
    // Use case: Setup LOFAR data emulator
    // Expect: Not to throw.
    try {
        EmulatorDriver emulator(new LofarUdpEmulator(_emulatorNode));
    }
    catch (const QString& error) {
        CPPUNIT_FAIL("Could not set up DataStreamingTest: " + error.toStdString());
    }
}


/**
 * @details
 * Run the data streaming tests with the direct streaming client
 */
void DataStreamingTest::test_streamingClient()
{
    Config config;
    config.setFromString(_adapterXML);
    Config::TreeAddress address;
    address << Config::NodeId("adapters", "");
    address << Config::NodeId("adapter", "test");
    ConfigNode configNode = config.get(address);
    LofarStreamDataClient* client = new LofarStreamDataClient(configNode, _dataTypes, &config);
    _testLofarDataClient(client);
    delete client;
}


/**
 * @details
 * Run the data streaming tests with the server client
 */
void DataStreamingTest::test_serverClient()
{
    // Set up a server to listen to the data
    // TODO
    // Configure the dataClient
    Config config;
    config.setFromString(_adapterXML);
    Config::TreeAddress address;
    address << Config::NodeId("adapters", "");
    address << Config::NodeId("adapter", "test");
    ConfigNode configNode = config.get(address);
    LofarServerClient* client = new LofarServerClient(configNode, _dataTypes, &config);

    // run the tests
    _testLofarDataClient(client);
    delete client;
}

/**
 * @details
 */
void DataStreamingTest::_testLofarDataClient(AbstractDataClient* client)
{
    // TODO : write tests to ensure data is being received OK
    // client->getData();
}

} // namespace lofar
} // namespace pelican
