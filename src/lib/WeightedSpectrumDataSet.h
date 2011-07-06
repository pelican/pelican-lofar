#ifndef WEIGHTEDSPECTRUMDATASET_H
#define WEIGHTEDSPECTRUMDATASET_H


/**
 * @file WeightedSpectrumDataSet.h
 */
#include "SpectrumDataSet.h"

namespace pelican {

namespace lofar {


/**
 * @class WeightedSpectrumDataSet
 *  
 * @brief
 *    Class that provides weights for each
 *    element in an associated SpectrumDataSet
 * @details
 *    The Associated dataset is modified 
 * 
 */

class WeightedSpectrumDataSet : public DataBlob
{
    public:
        WeightedSpectrumDataSet( SpectrumDataSet<float>* dat = 0 );
        ~WeightedSpectrumDataSet();
        void reset( SpectrumDataSet<float>* data );
        SpectrumDataSet<float>* dataSet() const { return _dataSet; };
        SpectrumDataSet<float>* weights() { return &_weights; };
        float rms() const;
        float mean() const;

    private:
        SpectrumDataSet<float>* _dataSet;
        SpectrumDataSet<float> _weights;
};
PELICAN_DECLARE_DATABLOB(WeightedSpectrumDataSet)

} // namespace lofar
} // namespace pelican
#endif // WEIGHTEDSPECTRUMDATASET_H 
