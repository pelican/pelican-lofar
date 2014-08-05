#ifndef LOFAR_DATA_SPLITTING_CHUNKER_TEST_H
#define LOFAR_DATA_SPLITTING_CHUNKER_TEST_H

/**
 * @file LofarChunkerTest.h
 */

#include <cppunit/extensions/HelperMacros.h>

#include "pelican/utility/Config.h"
#include "pelican/utility/ConfigNode.h"

namespace pelican {
namespace ampp {

/**
 * @class LofarDataSplittingChunkerTest
 *
 * @brief
 * Unit test for the LofarDataSplittingChunker class
 *
 * @details
 *
 */
class LofarDataSplittingChunkerTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE(LofarDataSplittingChunkerTest);
        CPPUNIT_TEST(test_normal_packets);
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_normal_packets();

    public:
        LofarDataSplittingChunkerTest();
        ~LofarDataSplittingChunkerTest();

    private:
        QString _serverXML;
        Config _config;
        ConfigNode _emulatorNode;

        QString _host;
        unsigned _port;
        QString _chunkType1;
        QString _chunkType2;

        // Packet dimensions.
        unsigned _nSamples;
        unsigned _nSubbands;
        unsigned _nSubbandsStream1;
        unsigned _nSubbandsStream2;
        unsigned _subbandStartStream1;
        unsigned _subbandEndStream1;
        unsigned _subbandStartStream2;
        unsigned _subbandEndStream2;
        unsigned _nPols;

        unsigned _nPackets;
        unsigned _sampleBits;
        unsigned _clock;
};

} // namespace ampp
} // namespace pelican
#endif // LOFAR_DATA_SPLITTING_CHUNKER_TEST_H
