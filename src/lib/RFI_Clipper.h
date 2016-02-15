#ifndef RFI_CLIPPER_H
#define RFI_CLIPPER_H


#include "pelican/modules/AbstractModule.h"
#include <vector>
#include <numeric>
#include <functional>
#include <vector>
#include "BandPass.h"
#include <boost/circular_buffer.hpp>
/**
 * @file RFI_Clipper.h
 */

namespace pelican {

namespace ampp {
    class SpectrumDataSetStokes;
    class WeightedSpectrumDataSet;
/**
 * @class RFI_Clipper
 *  
 * @brief
 *    Remove any Radio Frequency Interference by comparision with a bandpass filter
 * @details
 * 
 */

class RFI_Clipper : public AbstractModule
{
    public:
        RFI_Clipper( const ConfigNode& config );
        ~RFI_Clipper();

        void getLOFreqFromRedis();
        void run( WeightedSpectrumDataSet* weightedStokes );
        const BandPass& bandPass() const { return _bandPass; }; // return the BandPass Filter in use

    private:
        BinMap  _map;
        //        std::vector<float> _copyI;
        BandPass  _bandPass;
        bool _active;
        float _LOFreq;
        float _startFrequency;
        float _endFrequency;
        float _medianFromFile;
        float _rmsFromFile;
        float _crFactor, _srFactor; // scale factor for rejection (multiples of RMS)
	boost::circular_buffer<float> _meanBuffer, _rmsBuffer;
	float _rmsRunAve, _meanRunAve;
        int _current; // history pointer
        int _badSpectra;
        int _num, _numChunks;// number of values in history
        int _maxHistory; // max size of history buffer
// flag for removing median from each spectrum, equivalent to the zero-DMing technique
        int _zeroDMing; 
        std::vector<float> _lastGoodSpectrum;
};

PELICAN_DECLARE_MODULE(RFI_Clipper)
} // namespace ampp
} // namespace pelican
#endif // RFI_CLIPPER_H 
