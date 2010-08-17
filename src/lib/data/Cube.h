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
        T* _a;   // array access to cube data. a[i]

    public:
        Cube() : _nX(0), _nY(0), _nZ(0), _C(0), _a(0) {}

        Cube(unsigned nZ, unsigned nY, unsigned nX)
        : _nX(nX), _nY(nY), _nZ(nZ), _C(0), _a(0)
        {
            size_t size = _nZ * (_nY * (_nX * sizeof(T) + sizeof(T*)) + sizeof(T**));
            _C = (T***) malloc(size);
            unsigned rp = (_nZ * sizeof(T**)) / sizeof(T*);
            unsigned dp = (_nZ * sizeof(T**) + _nZ * _nY * sizeof(T*)) / sizeof(T);
            for (unsigned z = 0; z < _nZ; ++z) {
                _C[z] = (T**)_C + rp + z * _nY;
                for (unsigned y = 0; y < _nY; ++y) {
                    _C[z][y]  = (T*)_C + dp + _nX * (z * _nY + y);
                }
            }
            _a = (T*)_C + dp;
        }

        Cube(Cube<T>& c)
        {
            _nX = c._nX;
            _nY = c._nY;
            _nZ = c._nZ;
            size_t size = _nZ * (_nY * (_nX * sizeof(T) + sizeof(T*)) + sizeof(T**));
            _C = (T***) malloc(size);
            memcpy((void*)_C, (void*)c._C, size);
            // Re-construct the lookup table pointers (so they don't point to the old data!)
            unsigned rp = (_nZ * sizeof(T**)) / sizeof(T*);
            unsigned dp = (_nZ * sizeof(T**) + _nZ * _nY * sizeof(T*)) / sizeof(T);
            for (unsigned z = 0; z < _nZ; ++z) {
                _C[z] = (T**)_C + rp + z * _nY;
                for (unsigned y = 0; y < _nY; ++y) {
                    _C[z][y]  = (T*)_C + dp + _nX * (z * _nY + y);
                }
            }
            _a = (T*)_C + dp;
        }

        virtual ~Cube() { clear(); }

    public:
        bool empty() const
        { return (_nX == 0 || _nY == 0 || _nZ == 0) ? true : false; }

        void clear() {
            _nX = _nY = _nZ = 0;
            if (_C) { free(_C); _C = 0; }
            _a = 0;
        }

        void resize(unsigned nZ, unsigned nY, unsigned nX)
        {
            // Check if we need to resize
            if (nZ != 0 && nZ == _nZ && nY != 0 && nY == _nY && nX != 0 && nX == _nX)
                return;
            _nX = nX;
            _nY = nY;
            _nZ = nZ;
            size_t size = _nZ * (_nY * (_nX * sizeof(T) + sizeof(T*)) + sizeof(T**));
            _C = (T***) realloc(_C, size);
            unsigned rp = (_nZ * sizeof(T**)) / sizeof(T*);
            unsigned dp = (_nZ * sizeof(T**) + _nZ * _nY * sizeof(T*)) / sizeof(T);
            for (unsigned z = 0; z < _nZ; ++z) {
                _C[z] = (T**)_C + rp + z * _nY;
                for (unsigned y = 0; y < _nY; ++y) {
                    _C[z][y]  = (T*)_C + dp + _nX * (z * _nY + y);
                }
            }
            _a = (T*)_C + dp;
        }

        void resize(unsigned nZ, unsigned nY, unsigned nX, T value)
        {
            resize(nZ, nY, nX);
            for (unsigned i = 0; i < (nZ * nY * nX); ++i) _a[i] = value;
        }

        void print()
        {
            for (unsigned z = 0; z < _nZ; ++z) {
                for (unsigned y = 0; y < _nY; ++y) {
                    for (unsigned x = 0; x < _nX; ++x) {
                        std::cout << _C[z][y][x] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl << std::endl;
            }
        }


    public:
        const T*** ptr() const { return _C; }
        T*** ptr() { return _C; }

        const T* arrayPtr() const { return _a; }
        T* arrayPtr() { return _a; }

        unsigned size() const { return _nX * _nY * _nZ; }
        unsigned nX() const { return _nX; }
        unsigned nY() const { return _nY; }
        unsigned nZ() const { return _nZ; }

    public:
        T operator() (unsigned z, unsigned y, unsigned x) const { return _C[z][y][x]; }
        T& operator() (unsigned z, unsigned y, unsigned x) { return _C[z][y][x]; }

        const T** operator[] (unsigned z) const { return _C[z]; }
        T** operator[] (unsigned z) { return _C[z]; }

        Cube<T>& operator= (const Cube<T>& other)
        {
            if (this != &other) // protect against invalid self assignment.
            {
                clear();
                _nX = other._nX; _nY = other._nY; _nZ = other._nZ;
                size_t size = _nZ * (_nY * (_nX * sizeof(T) + sizeof(T*)) + sizeof(T**));
                _C = (T***) malloc(size);
                memcpy((void*)_C, (void*)other._C, size);
                // Re-construct the lookup table pointers (so they dont point to the old data!)
                unsigned rp = (_nZ * sizeof(T**)) / sizeof(T*);
                unsigned dp = (_nZ * sizeof(T**) + _nZ * _nY * sizeof(T*)) / sizeof(T);
                for (unsigned z = 0; z < _nZ; ++z) {
                    _C[z] = (T**)_C + rp + z * _nY;
                    for (unsigned y = 0; y < _nY; ++y) {
                        _C[z][y]  = (T*)_C + dp + _nX * (z * _nY + y);
                    }
                }
                _a = (T*)_C + dp;
            }
            // By convention always return *this.
            return *this;
        }
};


#endif // PELICAN_CUBE_H_
