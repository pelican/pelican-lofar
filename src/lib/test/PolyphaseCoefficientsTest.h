#ifndef POLYPHASE_COEFFICIENTS_TEST_H
#define POLYPHASE_COEFFICIENTS_TEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file PolyphaseCoefficientsTest.h
 */

/**
 * @class PolyphaseCoefficientsTest
 *
 * @brief
 * Unit testing class for the polyphase coefficients data blob.
 *
 * @details
 * Performs unit tests on the polyphase coefficients data blob object
 * using the CppUnit framework.
 */

namespace pelican {
namespace lofar {

class PolyphaseCoefficientsTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE(PolyphaseCoefficientsTest);
        CPPUNIT_TEST(test_accessorMethods);
        //CPPUNIT_TEST(test_loadCoeffFile);
        //CPPUNIT_TEST(test_generate);
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        /// Test accessor methods.
        void test_accessorMethods();
        /// Test loading a coeff file.
        void test_loadCoeffFile();
        /// Test generating coefficients.
        void test_generate();

    public:
        PolyphaseCoefficientsTest();
        ~PolyphaseCoefficientsTest();
};

} // namespace lofar
} // namespace pelican

#endif // POLYPHASE_COEFFICIENTS_TEST_H
