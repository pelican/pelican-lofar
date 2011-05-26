#ifndef BANDPASS_H
#define BANDPASS_H


#include "pelican/data/DataBlob.h"
#include "BinMap.h"
#include "Range.h"

#include <QVector>
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
        const QVector<float>& currentSet() { return _dataSets[_currentMapId]; };
        inline float intensityOfBin(unsigned int index) const {
            return _dataSets[_currentMapId][index];
        };
        inline float median() const { return _median[_currentMapId]; }
        inline float rms() const { return _rms[_currentMapId]; }
        // Mark channels to be killed (set to 0)
        void killChannel(unsigned int index);
        void killBand(float startFreq, float endFreq);
        // return true if bin has been killed
        bool filterBin( unsigned int i ) const;

    protected:
        float _evaluate(float) const; // calculate value of parameterised eqn
        void _zeroChannelsMap(const BinMap& map);

    private:
        int _nChannels;
        BinMap _primaryMap;
        BinMap _currentMap;
        int _currentMapId;
        int _primaryMapId;
        QVector<float> _params;
        float _deltaFreq;
        QHash<int, QVector<float> > _dataSets;
        QHash<int,float> _rms;
        QHash<int,float> _median;
        Range<float> _killed;
};

PELICAN_DECLARE_DATABLOB(BandPass)

} // namespace lofar
} // namespace pelican


#endif // BANDPASS_H 
