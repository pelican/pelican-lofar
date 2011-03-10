#include "BandPass.h"
#include "BinnedData.h"
#include "BinMap.h"
#include <cmath>


namespace pelican {

namespace lofar {


/**
 *@details BandPass 
 */
BandPass::BandPass()
    : DataBlob( "BandPass" )
{
}

/**
 *@details
 */
BandPass::~BandPass()
{
}

void BandPass::setData(const BinMap& map,const QVector<float>& params ) {
    _primaryMap = map;
    _params = params;
}

float BandPass::intensity(float frequency, const BinMap& map ) const
{
    // our ref data corresponds to a specific binning
    // where this is different we need to adjust the intensities
    if( ! _dataSets.contains(map) ) {
        // create the binned data with the
        // parameterised equation, scaled suitably
        BinnedData binnedData(map); 
        float scale = map.width()/_primaryMap.width();
        for( int i=0; i < map.numberBins(); ++i ) {
             binnedData.setBin(i, scale * _evaluate(frequency));
        }
        _dataSets.insert(map, binnedData);
    }
    int index = map.binIndex(frequency);
    return _dataSets[map][index];
}

float BandPass::_evaluate(float v) const
{
   float tot = 0.0;
   for(int i=0; i< _params.size(); ++i ) {
        tot += _params[i]*std::pow(v,i);
   }
   return tot;
}

} // namespace lofar
} // namespace pelican
