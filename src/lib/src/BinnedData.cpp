#include "BinnedData.h"

namespace pelican {
namespace lofar {

/**
 *@details BinnedData 
 */
BinnedData::BinnedData(const BinMap& map)
    : _map(map)
{
    //_bin.resize(_map.numberBins());
}

/**
 *@details
 */
BinnedData::~BinnedData()
{
}

BinnedData BinnedData::binAs(const BinMap& map) const
{
     // trivial case where binning is the same
     if( map.equals(_map) ) 
                return BinnedData(*this);

     // must rebin (ignores out of range)
     BinnedData r(map);
     float scale = _map.width()/map.width();
     int offset = (int)((map.startValue() - _map.startValue()) / _map.width());
     if(offset < 0 ) offset = 0; // assume all lower values equally weighted
     for( int i = 0; i < map.numberBins(); ++i )
     {
        int j = (int)(i*scale)+offset;
        if( j > _bin.size() ) 
                j = _bin.size();
        r.setBin( i, _bin[j]*scale );
     }
     return r;
}

void BinnedData::setBin(int binIndex, float value)
{
    _bin[binIndex] = value;
}

} // namespace lofar
} // namespace pelican
