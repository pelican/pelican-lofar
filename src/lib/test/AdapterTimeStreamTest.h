#ifndef ADAPTERTIMESTREAMTEST_H
#define ADAPTERTIMESTREAMTEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file AdapterTimeStreamTest
 */

namespace pelican {
namespace lofar {

/**
 * @class AdapterTimeStreamTest
 *
 * @brief
 * Unit tests for the AdapterTimeStreamTest class
 *
 * @details
 *
 */

class AdapterTimeStreamTest : public CppUnit::TestFixture
{
    public:
        AdapterTimeStreamTest() : CppUnit::TestFixture() {}
        ~AdapterTimeStreamTest() {}

    public:
        CPPUNIT_TEST_SUITE(AdapterTimeStreamTest);
        CPPUNIT_TEST(test_configuration);
        CPPUNIT_TEST(test_checkData);
        CPPUNIT_TEST(test_deserialise);
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        /// Method to test the adapter configuration.
        void test_configuration();
        /// Method to check data validation performed by the adapter.
        void test_checkData();
        /// Method to check deserialising a chunk of UDP packets.
        void test_deserialise();
};


} // namespace lofar
} // namespace pelican
#endif // ADAPTERTIMESTREAMTEST_H
