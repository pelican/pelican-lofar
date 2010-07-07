#ifndef SUBBAND_SPECTRA_TEST_H_
#define SUBBAND_SPECTRA_TEST_H_

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file SubbandSpectraTest.h
 */

/**
 * @class SubbandSpectraTest
 *
 * @brief
 * Unit testing class for the sub-band spectra data blob.
 *
 * @details
 */

namespace pelican {
namespace lofar {

class SubbandSpectraTest : public CppUnit::TestFixture
{
    public:
        SubbandSpectraTest() : CppUnit::TestFixture() {}
        ~SubbandSpectraTest() {}

    public:
        void setUp() {}
        void tearDown() {}

        /// Test accessor methods.
        void test_accessorMethods();
        void test_serialise_deserialise();

        CPPUNIT_TEST_SUITE(SubbandSpectraTest);
        CPPUNIT_TEST(test_accessorMethods);
        CPPUNIT_TEST(test_serialise_deserialise);
        CPPUNIT_TEST_SUITE_END();

    private:

};

} // namespace lofar
} // namespace pelican

#endif // SUBBAND_SPECTRA_TEST_H_
