#ifndef RFI_CLIPPER_H
#define RFI_CLIPPER_H


#include "pelican/modules/AbstractModule.h"
#include "BandPass.h"

/**
 * @file RFI_Clipper.h
 */

namespace pelican {

namespace lofar {
    class SpectrumDataSetStokes;
    class WeightedSpectrumDataSet;
/**
 * @class RFI_Clipper
 *  
 * @brief
 *    Remove any Radio Frequency Iinterference by comparision with a bandpass filter
 * @details
 * 
 */

class RFI_Clipper : public AbstractModule
{
    public:
        RFI_Clipper( const ConfigNode& config );
        ~RFI_Clipper();
        void run(SpectrumDataSetStokes* stokesI);
        void run(SpectrumDataSetStokes* stokesI, WeightedSpectrumDataSet* weights);
        const BandPass& bandPass() const { return _bandPass; }; // return the BandPass Filter in use

    private:
        BinMap  _map;
        BandPass  _bandPass;
        bool _active;
        float _startFrequency;
        float _endFrequency;
        float _rFactor; // scale factor for rejection (multiples of RMS)
};

    PELICAN_DECLARE_MODULE(RFI_Clipper)
} // namespace lofar
} // namespace pelican
#endif // RFI_CLIPPER_H 
