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
BandPass::~BandPass() {
}

void BandPass::setData(const BinMap& map,const QVector<float>& params ) {
    _primaryMap = map;
    _params = params;
     reBin( map );
}

void BandPass::setRMS(float rms) {
    _rms[_currentMap] = rms;
}

void BandPass::setMedian(float mean) {
    _mean[_currentMap] = mean;
}

void BandPass::reBin(const BinMap& map)
{
    _currentMap = map;
    if( ! _dataSets.contains(map) ) {
        BinnedData binnedData(map); 
        float scale = map.width()/_primaryMap.width();
        // scale the RMS and mean
        _rms[map]= _rms[_primaryMap] * std::sqrt( scale );
        // scale and set the intensities
        for( unsigned int i=0; i < map.numberBins(); ++i ) {
            binnedData.setBin( i, scale * _evaluate(map.binAssignmentNumber(i)));
        }
        _dataSets.insert(map, binnedData);
    }
}

float BandPass::intensity( float frequency ) const
{
    int index = _currentMap.binIndex(frequency);
    return _dataSets[_currentMap][index];
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
