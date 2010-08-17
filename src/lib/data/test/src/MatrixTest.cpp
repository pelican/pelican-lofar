#include "data/test/MatrixTest.h"
#include "data/Matrix.h"

#include <iostream>
#include <complex>

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION(MatrixTest);


/**
 * @details
 */
void MatrixTest::test_accessorMethods()
{
    unsigned nRows = 5, nColumns = 3;

    // Test Constructor.
    Matrix<int> m(nRows, nColumns);
    CPPUNIT_ASSERT_EQUAL(nRows, m.nRows());
    CPPUNIT_ASSERT_EQUAL(nRows, m.nY());
    CPPUNIT_ASSERT_EQUAL(nColumns, m.nColumns());
    CPPUNIT_ASSERT_EQUAL(nColumns, m.nX());

    // Test clear method.
    m.clear();
    CPPUNIT_ASSERT_EQUAL(unsigned(0), m.nRows());
    CPPUNIT_ASSERT_EQUAL(unsigned(0), m.nColumns());

    // Test resize method.
    nRows = 4, nColumns = 4;
    m.resize(nRows, nColumns);
    CPPUNIT_ASSERT_EQUAL(nRows, m.nRows());
    CPPUNIT_ASSERT_EQUAL(nColumns, m.nColumns());

    // Test resize and assign to value method.
    nRows = 7, nColumns = 7;
    m.clear();
    int value = -6;
    m.resize(nRows, nColumns, value);
    CPPUNIT_ASSERT_EQUAL(nRows, m.nRows());
    CPPUNIT_ASSERT_EQUAL(nColumns, m.nColumns());

    // Test pointer access.
    int** M = m.ptr();
    int* a = m.arrayPtr();
    int* row2 = m.rowPtr(2);
    CPPUNIT_ASSERT_EQUAL(&M[0][0], &a[0]);
    CPPUNIT_ASSERT_EQUAL(&M[2][0], &row2[0]);

    // Test operators [] and ().
    CPPUNIT_ASSERT_EQUAL(value, m[1][2]);
    CPPUNIT_ASSERT_EQUAL(value, m(1,2));

    // Test copy constructor.
    Matrix<int> u(m);
    CPPUNIT_ASSERT_EQUAL(nRows, u.nRows());
    CPPUNIT_ASSERT_EQUAL(nColumns, u.nColumns());
    CPPUNIT_ASSERT_EQUAL(value, u[1][2]);
    CPPUNIT_ASSERT_EQUAL(value, u(1,2));
}

} // namespace lofar
} // namespace pelican
