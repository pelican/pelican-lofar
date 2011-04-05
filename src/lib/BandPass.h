#ifndef BANDPASS_H
#define BANDPASS_H


#include "pelican/data/DataBlob.h"
#include "BinMap.h"
#include "Range.h"

#include <QVector>
#include <QMap>
#include <QHash>
#include <QPair>
#include <QSet>

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
        float startFrequency() const;
        float endFrequency() const;
        float intensity(float frequency) const;
        float intensityOfBin(unsigned int index) const;
        float median() const { return _median[_currentMap]; }
        float rms() const { return _rms[_currentMap]; }
        // Mark channels to be killed (set to 0)
        void killChannel(unsigned int index);
        void killBand(float startFreq, float endFreq);
        // return true if bin has been killed
        bool filterBin( unsigned int i );

    protected:
        float _evaluate(float) const; // calculate value of parameterised eqn
        void _zeroChannelsMap(const BinMap& map);

    private:
        int _nChannels;
        BinMap _primaryMap;
        BinMap _currentMap;
        QVector<float> _params;
        float _deltaFreq;
        QMap<BinMap, QVector<float> > _dataSets;
        QMap<BinMap,float> _rms;
        QMap<BinMap,float> _median;
        Range<float> _killed;
};

PELICAN_DECLARE_DATABLOB(BandPass)

} // namespace lofar
} // namespace pelican


#endif // BANDPASS_H 
