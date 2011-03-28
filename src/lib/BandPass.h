#ifndef BANDPASS_H
#define BANDPASS_H


#include "pelican/data/DataBlob.h"
#include "BinMap.h"

#include <QVector>
#include <QMap>
#include <QHash>

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
        void setRMS(float);
        void setMedian(float);
        void reBin(const BinMap& map);
        float intensity(float frequency) const;
        float intensityOfBin(unsigned index) const;
        float median() const { return _median[_currentMap]; }
        float rms() const { return _rms[_currentMap]; }

    protected:
        float _evaluate(float) const; // calculate value of parameterised eqn

    private:
        int _nChannels;
        BinMap _primaryMap;
        BinMap _currentMap;
        QVector<float> _params;
        float _deltaFreq;
        QMap<BinMap, QVector<float> > _dataSets;
        QMap<BinMap,float> _rms;
        QMap<BinMap,float> _median;
};

PELICAN_DECLARE_DATABLOB(BandPass)

} // namespace lofar
} // namespace pelican


#endif // BANDPASS_H 
