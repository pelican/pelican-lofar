#include "BandPass.h"
#include "BinMap.h"
#include "Range.h"
#include <cmath>
#include <algorithm>
#include <iostream>


namespace pelican {

namespace ampp {


/**
 *@details BandPass 
 */
BandPass::BandPass()
    : DataBlob( "BandPass" )
{
    _params.insert(0,0.0f);
}

BandPass::~BandPass() {
}

void BandPass::setData(const BinMap& map,const QVector<float>& params ) {
    _primaryMap = map;
    _primaryMapId = map.hash();
    _params = params;
    reBin( map );

    // calculate the median & mean for the primary map
    std::vector<float> copy;
    _mean[_primaryMapId] = 0;
    for( unsigned int i=0; i < map.numberBins(); ++i ) {
        _mean[_primaryMapId] += _dataSets[_primaryMapId][i];
        copy.push_back(_dataSets[_primaryMapId][i]);
    }
    _mean[_primaryMapId] /= map.numberBins();
    std::nth_element(copy.begin(), copy.begin()+copy.size()/2, copy.end());
    _median[_primaryMapId] = (float)*(copy.begin()+copy.size()/2);
}

void BandPass::setRMS(float rms) {
    _rms[_currentMapId] = rms;
}

void BandPass::setMedian(float median) {
    float delta = median - _median[_currentMapId];
    if( std::fabs(delta) > 0.0f ) {
        // set the new median and rescale the polynomial
        float scale = _currentMap.width()/_primaryMap.width();
        _median[_currentMapId] = median;
        _median[_primaryMapId] = median / scale;
        _params[0] += delta/scale;
        _mean[_currentMapId] += delta;
        _mean[_primaryMapId] += delta / scale;
        _dataSets.clear();
        _buildData(_currentMap, scale, delta);
    }
}

void BandPass::setMean(float mean) {
    float delta = mean - _mean[_currentMapId];
    if( std::fabs(delta) > 0.0f ) {
        // set the new median and rescale the polynomial
        float scale = _currentMap.width()/_primaryMap.width();
        _params[0] += delta/scale;
        _mean[_currentMapId] = mean;
        _mean[_primaryMapId] = mean / scale;
        _dataSets.clear();
        _buildData(_currentMap, scale, delta);
    }
}

void BandPass::resetMap()
{
    reBin(_primaryMap);
}

void BandPass::reBin(const BinMap& map)
{
    _currentMap = map;
    int mapId = map.hash();
    _currentMapId = mapId;
    if( ! _dataSets.contains(mapId) ) {
        double scale = map.width()/_primaryMap.width();
        // scale the RMS and median
        _rms[mapId]= _rms[_primaryMapId] * std::sqrt( 1.0/scale );
        _median[mapId] = _median[_primaryMapId] * scale;
        _mean[mapId] = _mean[_primaryMapId] * scale;
        // scale and set the intensities
        _buildData(_currentMap, scale, 0.0);
        //for( unsigned int i=0; i < map.numberBins(); ++i ) {
        //   _dataSets[mapId][i] = scale * _evaluate(map.binAssignmentNumber(i));
        //}
        //_zeroChannelsMap(map);
    }
}

void BandPass::_buildData(const BinMap& map, float scale, float /*offset*/) {
    int mapId = map.hash();
    _dataSets.insert(mapId, QVector<float>(map.numberBins()) );
    for( unsigned int i=0; i < map.numberBins(); ++i ) {
        _dataSets[mapId][i] = scale * _evaluate(map.binAssignmentNumber(i));
    }
    _zeroChannelsMap(map);
}

// set the dataset for a specified map to zero for all killed bands
void BandPass::_zeroChannelsMap(const BinMap& map)
{
     foreach( const Range<float>& r, _killed.subranges() ) {
         int min = map.binIndex(r.min());
         int max = map.binIndex(r.max());
         if( max < min ) { int tmp; tmp = max; max = min; min = tmp; };
         int mapId=map.hash();
         if(_dataSets[mapId].size() < max ) _dataSets[mapId].resize(max + 1);
         do {
             _dataSets[mapId][min] = 0.0;
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
    return _dataSets[_currentMapId][index];
}

void BandPass::killChannel(unsigned int index)
{
    killBand( _currentMap.binStart(index), _currentMap.binEnd(index) );
}

bool BandPass::filterBin( unsigned int index ) const
{
    return _dataSets[_currentMapId][index] < 0.0000001;
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

} // namespace ampp
} // namespace pelican
