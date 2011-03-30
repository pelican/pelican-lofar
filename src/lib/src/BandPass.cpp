#include "BandPass.h"
#include "BinMap.h"
#include <cmath>
#include <iostream>


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

void BandPass::setMedian(float median) {
    _median[_currentMap] = median;
}

void BandPass::reBin(const BinMap& map)
{
    _currentMap = map;
    if( ! _dataSets.contains(map) ) {
        _dataSets.insert(map, QVector<float>(map.numberBins()) );
        float scale = map.width()/_primaryMap.width();
        // scale the RMS and median
        _rms[map]= _rms[_primaryMap] * std::sqrt( 1.0/scale );
        _median[map] = _median[_primaryMap] * scale;
        // scale and set the intensities
        for( unsigned int i=0; i < map.numberBins(); ++i ) {
           _dataSets[map][i] = scale * _evaluate(map.binAssignmentNumber(i));
        }
    }
}

float BandPass::startFrequency() const
{
    return _currentMap.startValue();
}

float BandPass::endFrequency() const
{
    return _currentMap.endValue();
}

float BandPass::intensity( float frequency ) const
{
    int index = _currentMap.binIndex(frequency);
    return _dataSets[_currentMap][index];
}

float BandPass::intensityOfBin( unsigned index ) const
{
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
