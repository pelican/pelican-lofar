#include "WeightedSpectrumDataSet.h"

#include <numeric>

namespace pelican {

namespace lofar {


/**
 *@details WeightedSpectrumDataSet 
 */
WeightedSpectrumDataSet::WeightedSpectrumDataSet( SpectrumDataSet<float>* data )
   : DataBlob("WeightedSpectrumDataSet"), _dataSet(data)
{
     if( data ) {
         _weights.resize(*data);
         _weights.init(1.0);
     }
}

/**
 *@details
 */
WeightedSpectrumDataSet::~WeightedSpectrumDataSet()
{
}

void WeightedSpectrumDataSet::reset( SpectrumDataSet<float>* data )
{
     Q_ASSERT(data);
     _dataSet = data;
     _weights.resize(*data);
     _weights.init(1.0);
     _mean = 0.0f;
     _median = 0.0f;
     _rms = 0.0f;
}

void WeightedSpectrumDataSet::setRMS(float rms)
{
    _rms = rms;
}

void WeightedSpectrumDataSet::setMedian(float median)
{
    _median = median;
}

void WeightedSpectrumDataSet::setMean(float mean)
{
    _mean = mean;
}

float WeightedSpectrumDataSet::median() const
{
    return _median;
}

float WeightedSpectrumDataSet::rms() const
{
    return _rms;
}

float WeightedSpectrumDataSet::mean() const
{
    if( _mean ) return _mean;
    return std::accumulate(_dataSet->begin(),_dataSet->end(),0.0)/
           std::accumulate(_weights.begin(),_weights.end(),0.0);
}

} // namespace lofar
} // namespace pelican
