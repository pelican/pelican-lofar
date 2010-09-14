#ifndef SPECTRUM_DATA_SET_TEST_H_
#define SPECTRUM_DATA_SET_TEST_H_

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

class SpectrumDataSetTest : public CppUnit::TestFixture
{
    public:
        SpectrumDataSetTest() : CppUnit::TestFixture() {}
        ~SpectrumDataSetTest() {}

    public:
        void setUp() {}
        void tearDown() {}

        /// Test accessor methods.
        void test_accessorMethods();
	void test_serialise_deserialise();

        CPPUNIT_TEST_SUITE(SpectrumDataSetTest);
        //CPPUNIT_TEST(test_accessorMethods);
	//CPPUNIT_TEST(test_serialise_deserialise);
        CPPUNIT_TEST_SUITE_END();

    private:

};

} // namespace lofar
} // namespace pelican

#endif // SPECTRUM_DATA_SET_TEST_H_
