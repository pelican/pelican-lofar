#ifndef LOFAR_DATA_SPLITTING_CHUNKER_TEST_H
#define LOFAR_DATA_SPLITTING_CHUNKER_TEST_H

/**
 * @file LofarChunkerTest.h
 */

#include <cppunit/extensions/HelperMacros.h>

#include "pelican/utility/Config.h"
#include "pelican/utility/ConfigNode.h"

namespace pelican {
namespace lofar {

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
        unsigned _nPols;

        unsigned _nPackets;
        unsigned _sampleBits;
        unsigned _clock;
};

} // namespace lofar
} // namespace pelican
#endif // LOFAR_DATA_SPLITTING_CHUNKER_TEST_H
