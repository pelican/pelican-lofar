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
int BinMap::binIndex(double value) const
{
    return (int)(0.5 + ((value - _lower)/ _width ));
}

void BinMap::setStart(double start)
{
    _lower = start;
}

void BinMap::setBinWidth(double width)
{
    _width = width;
    _halfwidth = width/2.0;
}

void BinMap::setEnd(double end)
{
    setBinWidth( ( end - _lower )/_nBins );
}

double BinMap::binStart(unsigned int index) const
{
   return _lower + _width*index;
}

double BinMap::binEnd(unsigned int index) const
{
   return _lower + _width*(index+1);
}

double BinMap::binAssignmentNumber(int index) const
{
    return _lower + ( _width * index ) + _halfwidth;
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
