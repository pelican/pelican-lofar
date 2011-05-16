#include "WeightedSpectrumDataSet.h"

#include <numeric>

namespace pelican {

namespace lofar {


/**
 *@details WeightedSpectrumDataSet 
 */
WeightedSpectrumDataSet::WeightedSpectrumDataSet( SpectrumDataSet<float>* data )
   : _dataSet(data)
{
     _weights.resize(*data);
     _weights.init(1.0);
}

/**
 *@details
 */
WeightedSpectrumDataSet::~WeightedSpectrumDataSet()
{
}

float WeightedSpectrumDataSet::rms() const
{
   // TODO
   return std::accumulate(_weights.begin(),_weights.end(),0.0);
}

float WeightedSpectrumDataSet::mean() const
{
    return std::accumulate(_dataSet->begin(),_dataSet->end(),0.0)/
           std::accumulate(_weights.begin(),_weights.end(),0.0);
}

} // namespace lofar
} // namespace pelican
