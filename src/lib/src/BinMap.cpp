#include "BinMap.h"


namespace pelican {
namespace lofar {

/**
 *@details BinMap 
 */
BinMap::BinMap()
{
}

BinMap::BinMap(unsigned int numberOfBins)
        : _nBins(numberOfBins)
{
}

/**
 *@details
 */
BinMap::~BinMap()
{
}

// 0 - (_nBins-1) if value is in range
int BinMap::binIndex(float value) const
{
    return (int)((value - _lower)/ _width);
}

void BinMap::setStart(float start)
{
    _lower = start;
}

void BinMap::setBinWidth(float width)
{
    _width = width;
}

void BinMap::setEnd(float end)
{
    _width = ( end - _lower )/_nBins;
}

float BinMap::binAssignmentNumber(int index) const
{
    return _lower + ( _width * index );
}

bool BinMap::equals(const BinMap& map) const
{
    return (_lower == map._lower) && (_width == map._width);
}

bool operator==(const BinMap& m1, const BinMap& m2)
{
    return m1.equals(m2);
}

bool BinMap::operator<(const BinMap& map) const
{
     return _lower < map._lower || _nBins < map._nBins || _width < map._width;
}

} // namespace lofar
} // namespace pelican
