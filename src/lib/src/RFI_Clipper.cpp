#include "RFI_Clipper.h"
#include "SpectrumDataSet.h"
#include <QFile>
#include <QString>
#include "BandPassAdapter.h"
#include "BandPass.h"
#include "BinMap.h"
#include "pelican/utility/ConfigNode.h"

namespace pelican {
namespace lofar {
    /**
     *@details RFI_Clipper 
     */
    RFI_Clipper::RFI_Clipper( const ConfigNode& config )
        : AbstractModule( config ), _active(true), _rFactor(3.0)
    {
        if( config.hasAttribute("active") &&  config.getAttribute("active").toLower() == QString("false") ) {
            _active = false;
        }
        // read in any fixed file data
        QString file = config.getOption("BandPassData", "file", "");
        if( file != "" ) { 
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

    // RFI clipper to be used with Stokes-I out of Stokes Generator
    void RFI_Clipper::run(SpectrumDataSetStokes* stokesI)
    {
        if( _active ) {
            float* I;
            unsigned nSamples = stokesI->nTimeBlocks();
            unsigned nSubbands = stokesI->nSubbands();
            unsigned nChannels = stokesI->nChannels();
            unsigned nBins = nChannels * nSubbands;

            BinMap map( nBins );
            map.setStart( _startFrequency );
            map.setEnd( _endFrequency );
            _bandPass.reBin(map);

            //std::vector<float> copyI(nChannels * nS);
            std::vector<float> copyI(nBins);
//            std::vector<float> res(nSubbands);
//            std::vector<float> subbandMean(nSubbands);
//            std::vector<float> subbandMedian(nSubbands);
//            std::vector<float> copySM(nSubbands);
//            std::vector<float> subbandRMS(nSubbands);
//            std::vector<float> copySubbandRMS(nSubbands);
            //float medianOfMedians, medianOfRMS, sumI, sumI2 ;
//            float sumI;
            // create an ordered copy of the data
            for (unsigned t = 0; t < nSamples; ++t) {
                int bin = -1;
                for (unsigned s = 0; s < nSubbands; ++s) {
                    I = stokesI -> spectrumData(t, s, 0);
                    for (unsigned c = 0; c < nChannels; ++c) {
                        copyI[++bin]=I[c];
                    }
                }
            }
            // calculate the DC offset between bandpass description and current spectrum
            std::nth_element(copyI.begin(), copyI.begin()+copyI.size()/2, copyI.end());
            float median = (float)*(copyI.begin()+copyI.size()/2);
            float medianDelta = median - _bandPass.median();

            // readjust relative to median
            float margin = std::fabs(_rFactor * _bandPass.rms());
            float doublemargin = margin * 2.0;
            for (unsigned t = 0; t < nSamples; ++t) {
                int bin = -1;
 //               float* I;
                //float DCoffset = 0.0;
                /* first loop to find the DC offset between bandpass and data
                   for (unsigned s = 0; s < nSubbands; ++s) {
                   I = stokesI -> spectrumData(t, s, 0);
                   for (unsigned c = 0; c < nChannels; ++c) {
                   DCoffset += I[c] - _bandPass.intensityOfBin( ++bin );
                   }
                   }
                   DCoffset /= bin;
                   bin = -1; */
#pragma omp parallel for
                    for (unsigned s = 0; s < nSubbands; ++s) {
                        int bin = (s * nChannels) - 1;
                        float *I = stokesI -> spectrumData(t, s, 0);
                        for (unsigned c = 0; c < nChannels; ++c) {
                            if( _bandPass.filterBin( ++bin ) ) {
                                I[c] = 0.0;
                                continue;
                            }
                            float bandpass = _bandPass.intensityOfBin( bin );
                            float res = I[c] - medianDelta - bandpass;
                            //float res = I[c] - DCoffset - _bandPass.intensityOfBin( bin );
                            if ( res > margin || I[c] > bandpass + doublemargin) {
                                // I[c] = _bandPass.intensityOfBin( bin ) + medianDelta + margin;
                                //                       I[c] -= res;
                                I[c] = 0.0;
                            } 
                        }
                    }
            }
        }
    }

} // namespace lofar
} // namespace pelican
