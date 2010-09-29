#ifndef LOFARCHUNKERTEST_H
#define LOFARCHUNKERTEST_H

#include <cppunit/extensions/HelperMacros.h>
#include "pelican/utility/Config.h"
#include "pelican/utility/ConfigNode.h"

/**
 * @file LofarChunkerTest.h
 */

namespace pelican {
namespace lofar {

/**
 * @class LofarChunkerTest
 *
 * @brief
 * Unit test for the LofarChunker class
 *
 * @details
 *
 */
class LofarChunkerTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( LofarChunkerTest );
        CPPUNIT_TEST( test_normalPackets );
        CPPUNIT_TEST( test_lostPackets );
        CPPUNIT_TEST_SUITE_END( );

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_normalPackets();
        void test_lostPackets();

    public:
        LofarChunkerTest();
        ~LofarChunkerTest();

    private:

    private:
        QString _serverXML;
        Config _config;
        ConfigNode _emulatorNode;

        // Data Params
        int _subbandsPerPacket;
        int _samplesPerPacket;
        int _nrPolarisations;
        int _numPackets;
        int _nSamples;
        int _clock;
};

} // namespace lofar
} // namespace pelican

#endif
