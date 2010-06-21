#ifndef DATASTREAMINGTEST_H
#define DATASTREAMINGTEST_H

#include <QtCore/QString>
#include <cppunit/extensions/HelperMacros.h>
#include "pelican/core/DataTypes.h"
#include "pelican/utility/ConfigNode.h"

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
        ConfigNode _emulatorNode;
        int _subbandsPerPacket, _samplesPerPacket, _nrPolarisations;
        int _port, _numPackets, _interval;
        QString _hostname;
        QString _adapterXML;
        pelican::DataTypes _dataTypes;
};

} // namespace lofar
} // namespace pelican

#endif // DATASTREAMINGTEST_H
