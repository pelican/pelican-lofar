#include "BandPass.h"
#include "BinMap.h"
#include "Range.h"
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
        _zeroChannelsMap(map);
    }
}

// set the dataset for a specified map to zero for all killed bands
void BandPass::_zeroChannelsMap(const BinMap& map)
{
     foreach( const Range<float>& r, _killed.subranges() ) {
         int min = map.binIndex(r.min());
         int max = map.binIndex(r.max());
         if( max < min ) { int tmp; tmp = max; max = min; min = tmp; };
         do {
             _dataSets[map][min] = 0.0;
         } while( ++min <= max );
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

void BandPass::killChannel(unsigned int index)
{
    killBand( _currentMap.binStart(index), _currentMap.binEnd(index) );
}

bool BandPass::filterBin( unsigned int index ) const
{
    return _dataSets[_currentMap][index] < 0.0000001;
}

void BandPass::killBand( float start, float end)
{
    _killed = _killed + Range<float>(start,end);
    foreach( const BinMap& map, _dataSets.keys() ) {
        _zeroChannelsMap(map);
    }
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
