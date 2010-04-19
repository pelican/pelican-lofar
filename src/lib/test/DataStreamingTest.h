#ifndef DATASTREAMINGTEST_H
#define DATASTREAMINGTEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file DataStreamingTest.h
 */

namespace pelican {
    class AbstractDataClient;
}

namespace pelicanLofar {

/**
 * @class DataStreamingTest
 *  
 * @brief
 *   class to test th DataClients data streaming
 * @details
 * 
 */

class DataStreamingTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( DataStreamingTest );
        CPPUNIT_TEST( test_streamingClient );
        CPPUNIT_TEST( test_serverClient );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_streamingClient();
        void test_serverClient();

    public:
        DataStreamingTest(  );
        ~DataStreamingTest();

    private:
        void _testLofarDataClient(pelican::AbstractDataClient* client);
};

} // namespace pelicanLofar
#endif // DATASTREAMINGTEST_H 
