#ifndef LOFAR_DATA_CUBE_TEST_H_
#define LOFAR_DATA_CUBE_TEST_H_

#include <cppunit/extensions/HelperMacros.h>
#include "data/LofarDataCube.h"

/**
 * @file LofarDataCubeTest.h
 */

/**
 * @class LofarDataCubeTest
 *
 * @brief
 *
 * @details
 */

namespace pelican {
namespace lofar {

class LofarDataCubeTest : public CppUnit::TestFixture
{
    public:
        LofarDataCubeTest() : CppUnit::TestFixture() {}
        ~LofarDataCubeTest() {}

    public:
        void setUp() {}
        void tearDown() {}

        /// Test accessor methods.
        void test_accessorMethods();

        CPPUNIT_TEST_SUITE(LofarDataCubeTest);
        CPPUNIT_TEST(test_accessorMethods);
        CPPUNIT_TEST_SUITE_END();
};

} // namespace lofar
} // namespace pelican

#endif // LOFAR_DATA_CUBE_TEST_H_
