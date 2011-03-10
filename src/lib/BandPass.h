#ifndef BANDPASS_H
#define BANDPASS_H


#include "pelican/data/DataBlob.h"
#include "BinMap.h"

#include <QVector>
#include <QMap>

/**
 * @file BandPass.h
 */

namespace pelican {

namespace lofar {

/**
 * @class BandPass
 *  
 * @brief
 *    Interface to the stations bandpass
 * @details
 * 
 */
class BinnedData;

class BandPass : public DataBlob
{
    public:
        BandPass(  );
        ~BandPass();
        void setData(const BinMap&,const QVector<float>& params );
        float intensity(float frequency, const BinMap& b) const;

    private:
        float _evaluate(float) const; // calculate value of parameterised eqn
        int _nChannels;
        BinMap _primaryMap;
        QVector<float> _params;
        float _deltaFreq;
        mutable QMap<BinMap,BinnedData> _dataSets;
};

PELICAN_DECLARE_DATABLOB(BandPass)

} // namespace lofar
} // namespace pelican


#endif // BANDPASS_H 
