#ifndef PELICAN_MATRIX_H_
#define PELICAN_MATRIX_H_

/**
 * @file Matrix.h
 */

/**
 * TODO:
*  Iterators. (for rows, whole memory)
*  Identity matrix constructor?
*  << operator (for printing)
*  assignment operator (sort out)
*  row vs column major (needed?)
*  transpose
*  sort out copy constructor for (const Matrix)
*
*  TODO write a << operator to print matrix
*
*  TODO operators:
*  Assignment: Matrix& operator= (Matrix& m) { return m; }
*  Math: Addition / Subtraction / devision / multiplication.
*/

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

template <typename T> class Matrix
{
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
            for (unsigned y = 0; y < _nRows; ++y) {
                _M[y] = (T*)_M + dp + y * _nCols;
            }
            _a = (T*)_M + dp;
        }

        /// Copies the matrix
        Matrix(Matrix& m)
        {
            _nRows = m.nRows();
            _nCols = m.nColumns();
            size_t size = _nCols * _nRows * sizeof(T) + _nRows * sizeof(T*);
            _M = (T**) malloc(size);
            memcpy(_M, const_cast<T**>(m.ptr()), size);
            // Re-construct the lookup table pointers (so they dont point to the old data!)
            unsigned dp = _nRows * sizeof(T*) / sizeof(T);
            for (unsigned y = 0; y < _nRows; ++y) {
                _M[y] = (T*)_M + dp + y * _nCols;
            }
            _a = (T*)_M + dp;
        }

        /// Destroys the matrix cleaning up memory
        virtual ~Matrix()
        {
            if (_M != 0) {
                free(_M);
                _M = 0;
            }
        }

    public:

        // nRows = nY, nCols = nX
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

        bool empty() const { return (_nCols == 0 || _nRows == 0) ? true : false; }

        void clear() {
            _nCols = 0;
            _nRows = 0;
            if (_M) { free(_M); _M = 0; }
            _a = 0;
        }

    public:
        void scale(T value)
        {
            for (unsigned i = 0; i < _nRows * _nCols; ++i) _a[i] *= value;
        }

        void fliplr()
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

        void flipud()
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

        T min() const
        {
            T min = _a[0];
            for (unsigned i = 0; i < _nRows * _nCols; ++i)
                min = std::min<T>(_a[i], min);
            return min;
        }

        T max() const
        {
            T max = _a[0];
            for (unsigned i = 0; i < _nRows * _nCols; ++i)
                max = std::max<T>(_a[i], max);
            return max;
        }

        T sum() const
        {
            T sum = 0.0;
            for (unsigned i = 0; i < _nRows * _nCols; ++i)
                sum += _a[i];
            return sum;
        }

    public:
        T** ptr() { return _M; }
        const T** ptr() const { return _M; }

        T* rowPtr(unsigned row) { return &_M[row][0]; }
        const T* rowPtr(unsigned row) const { return &_M[row][0]; }

        T* arrayPtr() { return _a; }
        const T* arrayPtr() const { return _a; }

        unsigned size() const { return _nCols * _nRows; }
        unsigned nColumns() const { return _nCols; }
        unsigned nX() const { return _nCols; }
        unsigned nRows() const { return _nRows; }
        unsigned nY() const { return _nRows; }

    public:
        T operator() (unsigned row, unsigned col) const { return _M[row][col]; }
        T& operator() (unsigned row, unsigned col) { return _M[row][col]; }

        const T* operator[] (unsigned row) const { return _M[row]; }
        T* operator[] (unsigned row) { return _M[row]; }

    private:
        unsigned _nCols;  // size in the x direction
        unsigned _nRows;  // size in the y direction
        T** _M;
        T* _a;
};

#endif // PELICAN_MATRIX_H_
