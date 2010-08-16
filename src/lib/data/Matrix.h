#ifndef PELICAN_MATRIX_H_
#define PELICAN_MATRIX_H_

/**
 * @file Matrix.h
 */

/**
 * TODO:
 *  look at boost matrix instead: http://goo.gl/IdLW
 *
 *  Iterators. (for rows, whole memory)
 *  Idenity matrix
 *  << operator
 *  row vs column major ?
 *  sort out copy constructor for (const Matrix)
 *
 */

#include <cstdlib>
#include <cstring>

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
        Matrix(Matrix &m)
        {
            _nRows = m.nRows();
            _nCols = m.nColumns();
            size_t size = _nCols * _nRows * sizeof(T) + _nRows * sizeof(T*);
            _M = (T**) malloc(size);
            memcpy(_M, const_cast<T**>(m.ptr()), size);
            unsigned mp = _nRows * sizeof(T*) / sizeof(T);
            _a = (T*)_M + mp;
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
            unsigned mp = _nRows * sizeof(T*) / sizeof(T);
            for (unsigned y = 0; y < _nRows; ++y) {
                _M[y] = (T*)_M + mp + y * _nCols;
            }
            _a = (T*)_M + mp;
        }

        // nRows = nY, nCols = nX
        void resize(unsigned nRows, unsigned nCols, T value)
        {
            resize(nRows, nCols);
            for (unsigned j = 0; j < _nRows; j++) {
                for (unsigned i = 0; i < _nCols; i++) {
                    _M[j][i] = value;
                }
            }
        }

        void clear() {
            _nCols = 0;
            _nRows = 0;
            free(_M);
            _M = NULL;
            _a = NULL;
        }

public: // accessors

        T** ptr() { return _M; }
        const T** ptr() const { return _M; }

        T* arrayPtr() { return _a; }
        const T* arrayPtr() const { return _a; }

        T* rowPtr(unsigned row) { return &_M[row][0]; }
        const T* rowPtr(unsigned row) const { return &_M[row][0]; }

        unsigned nColumns() const { return _nCols; }
        unsigned nX() const { return _nCols; }
        unsigned nRows() const { return _nRows; }
        unsigned nY() const { return _nRows; }
        unsigned size() const { return (_nCols * _nRows); }

    public:
        T& operator() (unsigned row, unsigned col) {
            return _M[row][col];
        }

        T operator() (unsigned row, unsigned col) const {
            return _M[row][col];
        }

        T* operator[] (unsigned row) { return _M[row]; }

        const T* operator[] (unsigned row) const { return _M[row]; }

        // write a << operator to print matrix

        /// Assignement operator
//        Matrix& operator= (Matrix& m) { return m; }

    private:
        unsigned _nCols;  // size in the x direction
        unsigned _nRows;  // size in the y direction
        T** _M;
        T* _a;
};

#endif // PELICAN_MATRIX_H_
