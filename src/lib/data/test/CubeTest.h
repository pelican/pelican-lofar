#ifndef CUBE_TEST_H_
#define CUBE_TEST_H_

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

class CubeTest : public CppUnit::TestFixture
{
    public:
        CubeTest() : CppUnit::TestFixture() {}
        ~CubeTest() {}

    public:
        void setUp() {}
        void tearDown() {}

        /// Test accessor methods.
        void test_accessorMethods();

        CPPUNIT_TEST_SUITE(CubeTest);
        CPPUNIT_TEST(test_accessorMethods);
        CPPUNIT_TEST_SUITE_END();
};

} // namespace lofar
} // namespace pelican

#endif // CUBE_TEST_H_
