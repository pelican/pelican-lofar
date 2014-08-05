#include "WeightedSpectrumDataSet.h"

#include <numeric>

namespace pelican {

namespace ampp {


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
     _stats.reset();
     //_mean = 0.0f;
     //_median = 0.0f;
     //_rms = 0.0f;
}

void WeightedSpectrumDataSet::setRMS(float rms)
{
    _stats.setRMS(rms);
}

void WeightedSpectrumDataSet::setMedian(float median)
{
    //_median = median;
    _stats.setMedian(median);
}

void WeightedSpectrumDataSet::setMean(float mean)
{
    _stats.setMean(mean);
}

float WeightedSpectrumDataSet::median() const
{
    return _stats.median();
}

float WeightedSpectrumDataSet::rms() const
{
    return _stats.rms();
}

float WeightedSpectrumDataSet::mean() const
{
    float mean = _stats.mean();
    if( mean ) return mean;
    return std::accumulate(_dataSet->begin(),_dataSet->end(),0.0)/
           std::accumulate(_weights.begin(),_weights.end(),0.0);
}

const BlobStatistics& WeightedSpectrumDataSet::stats() const
{
    return _stats;
}

} // namespace ampp
} // namespace pelican
