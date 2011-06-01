#include "RFI_Clipper.h"
#include "SpectrumDataSet.h"
#include "WeightedSpectrumDataSet.h"
#include <QFile>
#include <QString>
#include "BandPassAdapter.h"
#include "BandPass.h"
#include "BinMap.h"
#include "pelican/utility/ConfigNode.h"
#include "pelican/utility/pelicanTimer.h"

namespace pelican {
namespace lofar {
/**
 *@details RFI_Clipper 
 */
RFI_Clipper::RFI_Clipper( const ConfigNode& config )
    : AbstractModule( config ), _active(true), _rFactor(3.0)
{
    if( config.hasAttribute("active") &&  
        config.getAttribute("active").toLower() == QString("false") ) {
        _active = false;
    }
    // read in any fixed file data
    QString file = config.getOption("BandPassData", "file", "");
    if( file != "" && _active ) { 
        if(! QFile::exists(file)) 
            throw(QString("RFI_Clipper: File \"" + file + "\" does not exist"));

        QFile dataFile(file);
        if( ! dataFile.open(QIODevice::ReadOnly | QIODevice::Text) )
            throw(QString("RFI_Clipper: Cannot open File \"" + file + "\""));
        BandPassAdapter adapter(config);
        dataFile.waitForReadyRead(-1);
        adapter.config(&_bandPass, dataFile.size());
        adapter.deserialise(&dataFile);
    }
    else {
        if( _active )
            throw(QString("RFI_Clipper: <BandPassData file=?> not defined"));
    }

    if( config.hasAttribute("rejectionFactor")  )
            _rFactor = config.getAttribute("rejectionFactor").toFloat();
    if( config.getOption("Band", "matching" ) == "true" ) {
        _startFrequency = _bandPass.startFrequency();
        _endFrequency = _bandPass.endFrequency();
    }
    else {
        if( _active ) {
            if( config.getOption("Band", "startFrequency" ) == "" ) {
                    throw(QString("RFI_Clipper: <Band startFrequency=?> not defined"));
            }
            _startFrequency = config.getOption("Band","startFrequency" ).toFloat();

            QString efreq = config.getOption("Band", "endFrequency" );
            if( efreq == "" )
                    throw(QString("RFI_Clipper: <Band endFrequency=?> not defined"));
            _endFrequency= efreq.toFloat();
        }
    }
}

/**
 *@details
 */
RFI_Clipper::~RFI_Clipper()
{
}

void RFI_Clipper::run( WeightedSpectrumDataSet* weightedStokes )
{
    if( _active ) {
        SpectrumDataSetStokes* stokesI = 
                static_cast<SpectrumDataSetStokes*>(weightedStokes->dataSet());
        SpectrumDataSet<float>* weights = weightedStokes->weights();
        float* I;
        unsigned nSamples = stokesI->nTimeBlocks();
        unsigned nSubbands = stokesI->nSubbands();
        unsigned nChannels = stokesI->nChannels();
        unsigned nPolarisations = stokesI->nPolarisations();
        unsigned nBins = nChannels * nSubbands;

        _map.reset( nBins );
        _map.setStart( _startFrequency );
        _map.setEnd( _endFrequency );
        _bandPass.reBin(_map);

        // create an ordered copy of the data
        _copyI.resize(nBins);
        for (unsigned t = 0; t < nSamples; ++t) {
            int bin = -1;
            for (unsigned s = 0; s < nSubbands; ++s) {
                I = stokesI -> spectrumData(t, s, 0);
                for (unsigned c = 0; c < nChannels; ++c) {
                    _copyI[++bin]=I[c];
                }
            }
        }

        // calculate the DC offset between bandpass description and current spectrum
        std::nth_element(_copyI.begin(), _copyI.begin()+_copyI.size()/2, _copyI.end());

        // --- .50 microseconds to here (10000 iterations) ----------------
        float median = (float)*(_copyI.begin()+_copyI.size()/2);
        float medianDelta = median - _bandPass.median();
        // readjust relative to median
        float margin = std::fabs(_rFactor * _bandPass.rms());
        I = stokesI->data();
        float *W = weights->data();
        for (unsigned t = 0; t < nSamples; ++t) {
#pragma omp parallel for
            for (unsigned s = 0; s < nSubbands; ++s) {
                int bin = (s * nChannels) - 1;
                long index = stokesI->index(s, nSubbands, 
                                            0, nPolarisations,
                                            t, nChannels ); 
                                            
                //float *I = stokesI -> spectrumData(t, s, 0);
                //float *W = weights -> spectrumData(t, s, 0);
          /* this If statement doubles the loop time :(
                        if( _bandPass.filterBin( ++bin ) ) {
                            I[index +c] = 0.0;
                            continue;
                        }
          */
                for (unsigned c = 0; c < nChannels; ++c) {
                    ++bin;
                    float bandpass = _bandPass.intensityOfBin( bin );
                    float res = I[index +c] - medianDelta - bandpass;
                    if ( res > margin ) {
                        W[index +c] = 0.0;
                    }
                    I[index +c] *= W[index +c];
                }
            }
            // loop takes from 7millsecs (first iteration) to 7 microsecs (over 1000 samples) 
        }
    }
}

// RFI clipper to be used with Stokes-I out of Stokes Generator
void RFI_Clipper::run(SpectrumDataSetStokes* stokesI)
{
    if( _active ) {
        float* I;
        unsigned nSamples = stokesI->nTimeBlocks();
        unsigned nSubbands = stokesI->nSubbands();
        unsigned nChannels = stokesI->nChannels();
        unsigned nPolarisations = stokesI->nPolarisations();
        unsigned nBins = nChannels * nSubbands;

        _map.reset( nBins );
        _map.setStart( _startFrequency );
        _map.setEnd( _endFrequency );
        _bandPass.reBin(_map);
        float bpMedian = _bandPass.median();
        float margin = std::fabs(_rFactor * _bandPass.rms());
        float doubleMargin = margin * 2.0;
        const QVector<float>& bandPass = _bandPass.currentSet();

        // create an ordered copy of the data
        _copyI.resize(nBins);
        for (unsigned t = 0; t < nSamples; ++t) {
            int bin = -1;
            for (unsigned s = 0; s < nSubbands; ++s) {
                I = stokesI -> spectrumData(t, s, 0);
                for (unsigned c = 0; c < nChannels; ++c) {
                    _copyI[++bin]=I[c];
                }
            }
            //}

            // calculate the DC offset between bandpass description and current spectrum
            std::nth_element(_copyI.begin(), _copyI.begin()+_copyI.size()/2, _copyI.end());

            // --- .50 microseconds to here (10000 iterations) ----------------
            float median = (float)*(_copyI.begin()+_copyI.size()/2);
            float medianDelta = median - bpMedian;
            // readjust relative to median
            I = stokesI->data();
            //for (unsigned t = 0; t < nSamples; ++t) {
#pragma omp parallel for
            for (unsigned s = 0; s < nSubbands; ++s) {
                int bin = (s * nChannels) - 1;
                long index = stokesI->index(s, nSubbands, 
                        0, nPolarisations,
                        t, nChannels ); 

                //float *I = stokesI -> spectrumData(t, s, 0);
                for (unsigned c = 0; c < nChannels; ++c) {
                    ++bin;
                    /* this If statement doubles the loop time :(
                       if( _bandPass.filterBin( ++bin ) ) {
                       I[index + c] = 0.0;
                       continue;
                       }
                       */
                    //float res = I[index + c] - medianDelta - bandPass[bin];
                    if ( medianDelta > doubleMargin || I[index + c] - medianDelta - bandPass[bin] > margin ) {
                        // I[c] = _bandPass.intensityOfBin( bin ) + medianDelta + margin;
                        //                       I[c] -= res;
                        I[index + c] = 0.0;
                    }
                }
            }
    }
    // loop takes from 7millsecs (first iteration) to 7 microsecs (over 1000 samples) 
    }
}

} // namespace lofar
} // namespace pelican
