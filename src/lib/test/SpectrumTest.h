#ifndef SPECTRUM_TEST_H_
#define SPECTRUM_TEST_H_

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file SpectrumTest.h
 */

/**
 * @class SpectrumTest
 *
 * @brief
 * Unit testing class for the spectrum data blob.
 *
 * @details
 * Performs unit tests on the spectrum data blob object
 * using the CppUnit framework.
 */

namespace pelican {
namespace lofar {

class SpectrumTest : public CppUnit::TestFixture
{
    public:
        SpectrumTest() : CppUnit::TestFixture() {}
        ~SpectrumTest() {}

    public:
        void setUp() {}
        void tearDown() {}

        /// Test accessor methods.
        void test_accessorMethods();

        CPPUNIT_TEST_SUITE(SpectrumTest);
        CPPUNIT_TEST(test_accessorMethods);
        CPPUNIT_TEST_SUITE_END();

    private:

};

} // namespace lofar
} // namespace pelican

#endif // SPECTRUM_TEST_H_
