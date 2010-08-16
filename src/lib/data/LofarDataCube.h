#ifndef LOFAR_DATA_CUBE_H_
#define LOFAR_DATA_CUBE_H_

/**
 * @file LofarDataCube.h
 */

#include <QtCore/QIODevice>
#include <QtCore/QSysInfo>

#include <vector>
#include <complex>
#include <iostream>

namespace pelican {
namespace lofar {

/**
 * @class
 *
 * @ingroup pelican_lofar
 *
 * @brief
 * Data cube class with the dim3 being the fastest varying dimension.
 *
 * @details
 */

template <class T> class LofarDataCube
{
    public:
        ///
        LofarDataCube() : _nDim1(0), _nDim2(0), _nDim3(0) {}

        ///
        LofarDataCube(unsigned nDim1, unsigned nDim2, unsigned nDim3)
        {
            resize(nDim1, nDim2, nDim3);
        }

        ///
        virtual ~LofarDataCube() {}

    public:
        ///
        void clear()
        {
            _cube.clear();
            _nDim1 = _nDim2 = _nDim3 = 0;
        }

        ///
        void resize(unsigned nDim1, unsigned nDim2, unsigned nDim3)
        {
            _nDim1 = nDim1;
            _nDim2 = nDim2;
            _nDim3 = nDim3;
            _cube.resize(_nDim1 * _nDim2 * _nDim3);
        }

        ///
        unsigned index(unsigned iDim1, unsigned iDim2, unsigned iDim3)
        {
            return _nDim3 * ( iDim1 * _nDim3 + iDim2) + iDim3;
        }

        ///
        unsigned size() const { return _cube.size(); }

        ///
        unsigned nDim1() const { return _nDim1; }
        unsigned nDim2() const { return _nDim2; }
        unsigned nDim3() const { return _nDim3; }

        unsigned nZ() const { return _nDim1; }
        unsigned nY() const { return _nDim2; }
        unsigned nX() const { return _nDim3; }


        ///
        T* dataPtr(unsigned iDim1, unsigned iDim2, unsigned iDim3)
        {
            unsigned i = index(iDim1, iDim2, iDim3);
            return (_cube.size() > 0 && i < _cube.size()) ? &_cube[i] : 0;
        }

        ///
        const T* dataPtr(unsigned iDim1, unsigned iDim2, unsigned iDim3) const
        {
            unsigned i = index(iDim1, iDim2, iDim3);
            return (_cube.size() > 0 && i < _cube.size()) ? &_cube[i] : 0;
        }

        ///
        T& data(unsigned iDim1, unsigned iDim2, unsigned iDim3)
        {
            unsigned i = index(iDim1, iDim2, iDim3);
            return _cube[i];
        }

        ///
        const T& data(unsigned iDim1, unsigned iDim2, unsigned iDim3) const
        {
            unsigned i = index(iDim1, iDim2, iDim3);
            return _cube[i];
        }

    private:
        unsigned _nDim1;
        unsigned _nDim2;
        unsigned _nDim3;
        std::vector<T> _cube;
};

}// namespace lofar
}// namespace pelican

#endif // LOFAR_DATA_CUBE_H_
