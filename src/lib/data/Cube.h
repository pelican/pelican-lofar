#ifndef PELICAN_CUBE_H_
#define PELICAN_CUBE_H_

#include <cstdlib>
#include <cstring>

template <typename T> class Cube
{
    private:
        unsigned _nX; // Fastest varying dimension.
        unsigned _nY; //
        unsigned _nZ; // Slowest varying dimension.
        T*** _C; // Cube C[z][y][x]
        //T** _M;  // Maxtrix access to cube M[z][y] = C[z][y]
        T* _a;   // array access to cube data. a[i]

    public:
        Cube() : _nX(0), _nY(0), _nZ(0), _C(0), _a(0) {}

        Cube(unsigned nZ, unsigned nY, unsigned nX)
        : _nX(nX), _nY(nY), _nZ(nZ), _C(0), _a(0)
        {
            size_t size = _nX * _nY * _nZ * sizeof(T);
            size += _nY * _nZ * sizeof(T*);
            size += _nZ * sizeof(T**);
            _C = (T***) malloc(size);

            unsigned dp = (_nZ * sizeof(T**) + _nZ * _nY * sizeof(T*)) / sizeof(T);
            unsigned mp = _nZ * sizeof(T**) / sizeof(T);

            for (unsigned z = 0; z < _nZ; ++z) {
                _C[z] = (T**)_C + mp + z * nY * nZ;
                for (unsigned y = 0; y < _nY; ++y) {
                    _C[z][y] = (T*)_C + dp + z * y * _nX;
                }
            }

            _a =(T*)_C + dp;
        }

        virtual ~Cube()
        {
            if (_C != 0) {
                free(_C);
                _C = 0;
            }
        }

    public:
        T*** ptr() { return _C; }
        const T*** ptr() const { return _C; }

};


#endif // PELICAN_CUBE_H_
