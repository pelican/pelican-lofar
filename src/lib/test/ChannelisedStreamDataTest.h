#ifndef CHANNELISEDSTREAMDATATEST_H
#define CHANNELISEDSTREAMDATATEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file ChannelisedStreamDataTest.h
 */

/**
 * @class ChannelisedStreamDataTest
 *
 * @brief
 * Unit testing class for the channelised stream data blob.
 *
 * @details
 * Performs unit tests on the channelised stream data blob object
 * using the CppUnit framework.
 */

namespace pelican {
namespace lofar {

class ChannelisedStreamDataTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( ChannelisedStreamDataTest );
        CPPUNIT_TEST( test_accessorMethods );
        CPPUNIT_TEST( test_serialise_deserialise );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        /// Test accessor methods.
        void test_accessorMethods();

        /// Test serialise and deserialise methods.
        void test_serialise_deserialise();

    public:
        ChannelisedStreamDataTest();
        ~ChannelisedStreamDataTest();

    private:
};

} // namespace lofar
} // namespace pelican

#endif // CHANNELISEDSTREAMDATATEST_H
