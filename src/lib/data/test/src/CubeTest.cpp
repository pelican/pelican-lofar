#include "data/test/CubeTest.h"
#include "data/Cube.h"

#include <iostream>
#include <complex>
#include <iomanip>

using std::cout;
using std::endl;
using std::hex;
using std::setw;
using std::left;

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION(CubeTest);


/**
 * @details
 */
void CubeTest::test_accessorMethods()
{
    cout << "CubeTest" << endl;
    unsigned nX = 3, nY = 3, nZ = 4;
    Cube<short> cube(nZ, nY, nX);
    short*** C = cube.ptr();

    for (unsigned z = 0; z < nZ; ++z) {
        for (unsigned y = 0; y < nY; ++y) {
            for (unsigned x = 0; x < nX; ++x) {
                cout << hex << &C[z][y][x] << " ";
            }
            cout << endl;
        }
        cout << endl << endl;
    }




}

} // namespace lofar
} // namespace pelican
