#include "data/test/LofarDataCubeTest.h"
#include "data/LofarDataCube.h"

#include <iostream>
#include <complex>

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION(LofarDataCubeTest);


/**
 * @details
 */
void LofarDataCubeTest::test_accessorMethods()
{
    unsigned nX = 4, nY = 3, nZ = 2;

    LofarDataCube<int> c(nZ, nY, nX);
    CPPUNIT_ASSERT_EQUAL(nZ, c.nDim1());
    CPPUNIT_ASSERT_EQUAL(nY, c.nDim2());
    CPPUNIT_ASSERT_EQUAL(nX, c.nDim3());

    CPPUNIT_ASSERT_EQUAL(nZ, c.nZ());
    CPPUNIT_ASSERT_EQUAL(nY, c.nY());
    CPPUNIT_ASSERT_EQUAL(nX, c.nX());

    CPPUNIT_ASSERT_EQUAL(nX * nY * nZ, c.size());

    c.clear();
    CPPUNIT_ASSERT_EQUAL(unsigned(0), c.nDim1());
    CPPUNIT_ASSERT_EQUAL(unsigned(0), c.nDim2());
    CPPUNIT_ASSERT_EQUAL(unsigned(0), c.nDim3());

    c.resize(nZ, nY, nX);
    CPPUNIT_ASSERT_EQUAL(nZ, c.nDim1());
    CPPUNIT_ASSERT_EQUAL(nY, c.nDim2());
    CPPUNIT_ASSERT_EQUAL(nX, c.nDim3());

    int value = -1;
    c.data(0, 0, 0) = value;
    CPPUNIT_ASSERT_EQUAL(value, *c.dataPtr(0, 0, 0));
}

} // namespace lofar
} // namespace pelican
