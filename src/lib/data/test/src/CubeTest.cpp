#include "data/test/CubeTest.h"
#include "data/Cube.h"

#include <iostream>
#include <complex>
#include <iomanip>

using std::cout;
using std::endl;
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
    unsigned nX = 2, nY = 2, nZ = 2;
    Cube<char> cube(nZ, nY, nX);
    char*** C = cube.ptr();

    for (unsigned z = 0; z < nZ; ++z) {
        for (unsigned y = 0; y < nY; ++y) {
            for (unsigned x = 0; x < nX; ++x) {
                cout << (void*)&C[z][y][x] << " ";
            }
            cout << endl;
        }
        cout << endl << endl;
    }

    for (unsigned z = 0; z < nZ; ++z) {
        for (unsigned y = 0; y < nY; ++y) {
            for (unsigned x = 0; x < nX; ++x) {
                cout << (void*)&cube[z][y][x] << " ";
            }
            cout << endl;
        }
        cout << endl << endl;
    }

    cube.clear();
    CPPUNIT_ASSERT_EQUAL(true, cube.empty());

    nX = 3, nY = 3, nZ = 2;
    cube.resize(nZ, nY, nX);
    for (unsigned z = 0; z < nZ; ++z) {
        for (unsigned y = 0; y < nY; ++y) {
            for (unsigned x = 0; x < nX; ++x) {
                cout << (void*)&cube[z][y][x] << " ";
            }
            cout << endl;
        }
        cout << endl << endl;
    }

    for (unsigned z = 0; z < nZ; ++z) {
        for (unsigned y = 0; y < nY; ++y) {
            for (unsigned x = 0; x < nX; ++x) {
                cout << (void*)&cube(z,y,x) << " ";
            }
            cout << endl;
        }
        cout << endl << endl;
    }

    nX = 3, nY = 2, nZ = 2;
    cube.resize(nZ, nY, nX, 'a');
    for (unsigned z = 0; z < nZ; ++z) {
        for (unsigned y = 0; y < nY; ++y) {
            for (unsigned x = 0; x < nX; ++x) {
                cout << cube[z][y][x] << " ";
            }
            cout << endl;
        }
        cout << endl << endl;
    }

    Cube<char> cube2;
    nX = 2, nY = 2, nZ = 2;
    cube2.resize(nZ, nY, nX, 'b');
    Cube<char> cube3(cube2);

    for (unsigned z = 0; z < nZ; ++z) {
        for (unsigned y = 0; y < nY; ++y) {
            for (unsigned x = 0; x < nX; ++x) {
                cout << cube2[z][y][x] << " ";
            }
            cout << endl;
        }
        cout << endl << endl;
    }

    cube3[0][1][0] = 'c';
    for (unsigned z = 0; z < nZ; ++z) {
        for (unsigned y = 0; y < nY; ++y) {
            for (unsigned x = 0; x < nX; ++x) {
                cout << cube3[z][y][x] << " ";
            }
            cout << endl;
        }
        cout << endl << endl;
    }
}

} // namespace lofar
} // namespace pelican
