#ifndef BINNEDDATA_H
#define BINNEDDATA_H
#include <QVector>


/**
 * @file BinnedData.h
 */
#include "BinMap.h"

namespace pelican {

namespace lofar {

/**
 * @class BinnedData
 *  
 * @brief
 *   Very Simple class to represent 1D binned floating point data
 * @details
 * 
 */

class BinnedData
{
    public:
        BinnedData(const BinMap& map);
        BinnedData() {};
        ~BinnedData();
        BinnedData binAs(const BinMap&) const;
        void setBin(int binIndex, float value);
        inline float operator[](int i) const { return _bin[i]; };

    private:
        BinMap  _map;
        QVector<float> _bin;
};

} // namespace lofar
} // namespace pelican
#endif // BINNEDDATA_H 
