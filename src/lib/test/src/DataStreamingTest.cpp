#include "DataStreamingTest.h"
#include "LofarStreamDataClient.h"
#include "pelican/core/AbstractDataClient.h"
#include "pelican/utility/memCheck.h"

namespace pelicanLofar {

CPPUNIT_TEST_SUITE_REGISTRATION( DataStreamingTest );
/**
 *@details DataStreamingTest 
 */
DataStreamingTest::DataStreamingTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
DataStreamingTest::~DataStreamingTest()
{
}

void DataStreamingTest::setUp()
{
    // TODO
    // a) set up the lofar data emulator in a seperate thread
}

void DataStreamingTest::tearDown()
{
}

/*
 * Run the data streaming tests with the direct streaing client
 */
void DataStreamingTest::test_streamingClient()
{
    //LofarStreamDataClient* client = new LofarStreamDataClient();
    //_testLofarDataClient(client);
    //delete client;
}

/*
 * Run the data streaming tests with the server client
 */
void DataStreamingTest::test_serverClient()
{
    //LofarServerClient* client = new LofarServerClient();
    // Set up a server to listen to the data
    //_testLofarDataClient(client);
    //delete client;
}

void _testLofarDataClient(pelican::AbstractDataClient* client)
{
    // TODO : write tests to ensure data is being received OK
    // client->getData();
}

} // namespace pelicanLofar
