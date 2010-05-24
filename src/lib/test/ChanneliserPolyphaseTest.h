#ifndef CHANNELISERPOLYPHASETEST_H
#define CHANNELISERPOLYPHASETEST_H

#include <cppunit/extensions/HelperMacros.h>
#include "pelican/utility/ConfigNode.h"

/**
 * @file ChanneliserPolyphaseTest.h
 */

namespace pelican {
namespace lofar {

/**
 * @class ChanneliserPolyphaseTest
 *
 * @brief
 *
 * @details
 * Performs unit tests on the polyphase channeliser module using the
 * CppUnit framework.
 */

class ChanneliserPolyphaseTest : public CppUnit::TestFixture
{
    public:
        ChanneliserPolyphaseTest() : CppUnit::TestFixture() {};
        ~ChanneliserPolyphaseTest() {}

    public:
        CPPUNIT_TEST_SUITE(ChanneliserPolyphaseTest);
        CPPUNIT_TEST(test_configuration);
        CPPUNIT_TEST(test_updateBuffer);
        CPPUNIT_TEST(test_filter);
        CPPUNIT_TEST(test_fft);
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        /// Test module configuration.
        void test_configuration();
        void test_updateBuffer();
        void test_filter();
        void test_fft();

    private:
        void _setupConfig(ConfigNode& config, const unsigned nChannels,
        		const unsigned nTaps, const unsigned nSubbands,
        		const QString coeffFile);

};

} // namespace lofar
} // namespace pelican

#endif // CHANNELISERPOLYPHASETEST_H
