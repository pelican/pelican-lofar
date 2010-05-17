#ifndef CHANNELISERPOLYPHASETEST_H
#define CHANNELISERPOLYPHASETEST_H

#include <cppunit/extensions/HelperMacros.h>

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
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        /// Test module configuration.
        void test_configuration();
};

} // namespace lofar
} // namespace pelican

#endif // CHANNELISERPOLYPHASETEST_H
