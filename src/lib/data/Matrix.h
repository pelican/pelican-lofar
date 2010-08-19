#ifndef PELICAN_MATRIX_H_
#define PELICAN_MATRIX_H_

/**
 * @file Matrix.h
 */

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cassert>

namespace pelican {
namespace lofar {

template <typename T> class Matrix
{
    private:
        unsigned _nCols;  // Fastest varying dimension (x)
        unsigned _nRows;  // Slower varying dimension (y)
        T** _M; // Matrix pointer _M[y][x].
        T* _a;  // array pointer. _a[y * _nCols + x]

    public:
        /// Constructs an empty matrix
        Matrix() : _nCols(0), _nRows(0), _M(0), _a(0) {};

        /// Constructs a matrix of the specified size
        Matrix(unsigned nRows, unsigned nCols)
        : _nCols(nCols), _nRows(nRows), _M(0), _a(0)
        {
            size_t size = _nCols * _nRows * sizeof(T) + _nRows * sizeof(T*);
            _M = (T**) malloc(size);
            unsigned dp = _nRows * sizeof(T*) / sizeof(T);
            for (unsigned y = 0; y < _nRows; y++) {
                _M[y] = (T*)_M + dp + y * _nCols;
            }
            _a = (T*)_M + dp;
        }

        /// Copies the matrix
        Matrix(Matrix& m)
        {
            _nRows = m._nRows;
            _nCols = m._nCols;
            size_t size = _nCols * _nRows * sizeof(T) + _nRows * sizeof(T*);
            _M = (T**) malloc(size);
            memcpy((void*)_M, (void*)m._M, size);
            // Re-construct the lookup table pointers
            // (so they don't point to the old data!)
            unsigned dp = _nRows * sizeof(T*) / sizeof(T);
            for (unsigned y = 0; y < _nRows; ++y) {
                _M[y] = (T*)_M + dp + y * _nCols;
            }
            _a = (T*)_M + dp;
        }

        /// Destroys the matrix cleaning up memory.
        virtual ~Matrix() { clear(); }

    public:
        void resize(unsigned nRows, unsigned nCols)
        {
            // check if we need to resize
            if (nRows != 0 && nRows == _nRows && nCols != 0 && nCols == _nCols)
                return;
            _nRows = nRows;
            _nCols = nCols;
            size_t size = _nCols * _nRows * sizeof(T) + _nRows * sizeof(T*);
            _M = (T**) realloc(_M, size);
            unsigned dp = _nRows * sizeof(T*) / sizeof(T);
            for (unsigned y = 0; y < _nRows; ++y) {
                _M[y] = (T*)_M + dp + y * _nCols;
            }
            _a = (T*)_M + dp;
        }

        // nRows = nY, nCols = nX
        void resize(unsigned nRows, unsigned nCols, T value)
        {
            resize(nRows, nCols);
            for (unsigned i = 0; i < _nRows * _nCols; ++i) _a[i] = value;
        }

        void fill(T value)
        {
            for (unsigned i = 0; i < _nRows * _nCols; ++i) _a[i] = value;
        }

        bool empty() const { return (_nCols == 0 || _nRows == 0) ? true : false; }

        void clear()
        {
            _nRows = _nCols = 0;
            if (_M) { free(_M); _M = 0; }
            _a = 0;
        }

        // in bytes
        size_t mem()
        {
            return _nCols * _nRows * sizeof(T) + _nRows * sizeof(T*);
        }

    public: // Accessor methods.
        unsigned size() const { return _nCols * _nRows; }
        unsigned nColumns() const { return _nCols; }
        unsigned nX() const { return _nCols; }
        unsigned nRows() const { return _nRows; }
        unsigned nY() const { return _nRows; }

        Matrix& matrix() { return *this; }
        const Matrix& matrix() const { return *this; }

        T * * ptr() { return _M; }
        T const * const * ptr() const { return _M; }

        T const * rowPtr(unsigned row) const { return _M[row]; }
        T * rowPtr(unsigned row) { return _M[row]; }

        T * arrayPtr() { return _a; }
        const T* arrayPtr() const { return _a; }

        T operator () (unsigned row, unsigned col) const { return _M[row][col]; }
        T& operator () (unsigned row, unsigned col) { return _M[row][col]; }

        T const * operator [] (unsigned row) const { return _M[row]; }
        T * operator [] (unsigned row) { return _M[row]; }


    public:
        /// Assignment operator (M1 = M2).
        Matrix& operator = (const Matrix& other)
        {
            if (this != &other)
            {
                clear(); // this can be faster (don't need to clear always see vector header)
                _nCols = other._nCols; _nRows = other._nRows;
                size_t size = _nCols * _nRows * sizeof(T) + _nRows * sizeof(T*);
                _M = (T**) malloc(size);
                memcpy((void*)_M, (void*)other._M, size);
                // Re-construct the lookup table pointers
                // (so they don't point to the old data!)
                unsigned dp = _nRows * sizeof(T*) / sizeof(T);
                for (unsigned y = 0; y < _nRows; ++y) {
                    _M[y] = (T*)_M + dp + y * _nCols;
                }
                _a = (T*)_M + dp;
            }
            return *this;
        }

        Matrix& operator *= (const T c)
        {
            for (unsigned i = 0; i < _nRows * _nCols; ++i) _a[i] *= c;
            return *this;
        }

        Matrix& operator /= (const T c)
        {
            for (unsigned i = 0; i < _nRows * _nCols; ++i) _a[i] /= c;
            return *this;
        }

        // Addition. (M = M1 + M2)
        Matrix operator + (const Matrix& m) const
        {
            Matrix __M(_nRows, _nCols);
            T* a = __M.arrayPtr();
            for (unsigned i = 0; i < _nRows * _nCols; ++i) a[i] = _a[i] + m._a[i];
            return __M;
        }

        // Addition assignment (M += M1)
        Matrix& operator += (const Matrix& m)
        {
            for (unsigned i = 0; i < _nRows * _nCols; ++i) _a[i] += m._a[i];
            return *this;
        }

        Matrix& operator += (const T c)
        {
            for (unsigned i = 0; i < _nRows * _nCols; ++i) _a[i] += c;
            return *this;
        }

        Matrix& operator -= (const T c)
        {
            for (unsigned i = 0; i < _nRows * _nCols; ++i) _a[i] -= c;
            return *this;
        }

        // Subtraction. (M = M1 - M2)
        Matrix operator - (const Matrix& m) const
        {
            Matrix<T> __M(_nRows, _nCols);
            T* a = __M.arrayPtr();
            for (unsigned i = 0; i < _nRows * _nCols; ++i) a[i] = _a[i] - m._a[i];
            return __M;
        }

        // Subtraction assignment (M -= M1)
        Matrix& operator -= (const Matrix& m)
        {
            for (unsigned i = 0; i < _nRows * _nCols; ++i) _a[i] -= _a[i];
            return *this;
        }


        // Unitary minus. (M = -M);
        Matrix operator - ()
        {
            Matrix<T> __M(_nRows, _nCols);
            T* a = __M.arrayPtr();
            for (unsigned i = 0; i < _nRows * _nCols; ++i) a[i] = -_a[i];
            return __M;
        }

    public:
        /// Flip left-right.
        void fliplr();

        /// Flip up-down.
        void flipud();

        /// Return the minimum of the matrix.
        T min() const;

        /// Return the maximum of the matrix.
        T max() const;

        /// Return the sum of all element of the the matrix.
        T sum() const;
};



//
//------------------------------------------------------------------------------
// Inline method/function definitions.
//

template <typename T>
inline
void Matrix<T>::fliplr()
{
    size_t rowSize = _nCols * sizeof(T);
    T* tempRow = (T*) malloc(rowSize);
    for (unsigned y = 0; y < _nRows; ++y) {
        for (unsigned x = 0; x < _nCols; ++x) {
            tempRow[x] = _M[y][_nCols - x - 1];
        }
        memcpy(_M[y], tempRow, rowSize);
    }
    free (tempRow);
}


template <typename T>
inline
void Matrix<T>::flipud()
{
    size_t rowSize = _nCols * sizeof(T);
    T* tempRow = (T*) malloc(rowSize);
    unsigned yDest = 0;
    for (unsigned y = 0; y < floor(_nRows / 2); ++y) {
        yDest = _nRows - y - 1;
        memcpy(tempRow, _M[y], rowSize);
        memcpy(_M[y], _M[yDest], rowSize);
        memcpy(_M[yDest], tempRow, rowSize);
    }
    free (tempRow);
}

template <typename T>
inline
T Matrix<T>::min() const
{
    T min = _a[0];
    for (unsigned i = 0; i < _nRows * _nCols; ++i)
        min = std::min<T>(_a[i], min);
    return min;
}

template <typename T>
inline
T Matrix<T>::max() const
{
    T max = _a[0];
    for (unsigned i = 0; i < _nRows * _nCols; ++i)
        max = std::max<T>(_a[i], max);
    return max;
}

template <typename T>
inline
T Matrix<T>::sum() const
{
    T sum = 0.0;
    for (unsigned i = 0; i < _nRows * _nCols; ++i)
        sum += _a[i];
    return sum;
}


} // namespace lofar
} // namespace pelican
#endif // PELICAN_MATRIX_H_
