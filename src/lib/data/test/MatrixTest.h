#ifndef MATRIX_TEST_H_
#define MATRIX_TEST_H_

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file MatrixTest.h
 */

/**
 * @class MatrixTest
 *
 * @brief
 *
 * @details
 */

namespace pelican {
namespace lofar {

class MatrixTest : public CppUnit::TestFixture
{
    public:
        MatrixTest() : CppUnit::TestFixture() {}
        ~MatrixTest() {}

    public:
        void setUp() {}
        void tearDown() {}

        /// Test accessor methods.
        void test_accessorMethods();

        CPPUNIT_TEST_SUITE(MatrixTest);
        CPPUNIT_TEST(test_accessorMethods);
        CPPUNIT_TEST_SUITE_END();
};

} // namespace lofar
} // namespace pelican

#endif // MATRIX_TEST_H_
