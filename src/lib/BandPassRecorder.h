#ifndef BANDPASSRECORDER_H
#define BANDPASSRECORDER_H


#include "pelican/core/AbstractModule.h"
#include <vector>

/**
 * @file BandPassRecorder.h
 */

namespace pelican {

namespace ampp {
class SpectrumDataSetStokes;
class BandPass;
class BinMap;

/**
 * @class BandPassRecorder
 *  
 * @brief
 *    Measure a suitable BandPass from an incoming data stream
 * @details
 * 
 */

class BandPassRecorder : public AbstractModule
{
    public:
        BandPassRecorder( const ConfigNode& config );
        ~BandPassRecorder();
        // take it data and process it to fill in the BandPass object
        // returns false if more data is required, true if complete
        bool run( SpectrumDataSetStokes* stokesI, BandPass* bp );

    protected:
        // resets the working variables
        void _reset(const BinMap&);
        // generate a trial fitting function on the integrated data
        void _polyFit(float* x, int nDataPoints );
        // evaluate the current value for the specified freq
        float _theFit(float) const;

        int sgels( int n, int m, int nrhs, 
                   float *A, int lda, float *B, int ldb, 
                   float *workSpace, int* work );


    private:
        std::vector<float> _sum;
        unsigned long _totalSamples;
        unsigned long _requiredSamples;
        float _startFrequency;
        float _endFrequency;
        int _polyDegree;
        std::vector<float> _fit;
        std::vector<float> _freq;
        std::vector<float> _freqMatrix;
        std::vector<float> _valueMatrix;
};

PELICAN_DECLARE_MODULE(BandPassRecorder)

} // namespace ampp
} // namespace pelican
#endif // BANDPASSRECORDER_H 
