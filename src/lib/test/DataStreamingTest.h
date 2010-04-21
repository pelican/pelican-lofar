#ifndef DATASTREAMINGTEST_H
#define DATASTREAMINGTEST_H

#include <QString>
#include <cppunit/extensions/HelperMacros.h>
#include "LofarDataGenerator.h"
#include "pelican/core/DataTypes.h"

/**
 * @file DataStreamingTest.h
 */

namespace pelican {

class AbstractDataClient;

namespace lofar {

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
//        CPPUNIT_TEST( test_streamingClient );
//        CPPUNIT_TEST( test_serverClient );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_setupGenerator();
        void test_streamingClient();
        void test_serverClient();

    public:
        DataStreamingTest();
        ~DataStreamingTest();

    private:
        void _testLofarDataClient(pelican::AbstractDataClient* client);

    private:
        // Test parameters
        LofarDataGenerator dataGenerator;
        int subbandsPerPacket, samplesPerPacket, nrPolarisations;
        int port, numPackets, usecDelay;
        char hostname[20];
        QString adapterXML;
        pelican::DataTypes dataTypes;

};

} // namespace lofar
} // namespace pelican

#endif // DATASTREAMINGTEST_H
