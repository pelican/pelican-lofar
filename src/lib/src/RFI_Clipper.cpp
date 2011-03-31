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
            // calculate the mean of all subbands and channels
            std::nth_element(copyI.begin(), copyI.begin()+copyI.size()/2, copyI.end());
            float median = (float)*(copyI.begin()+copyI.size()/2);
            float medianDelta = median - _bandPass.median();

            // readjust relative to median
            float margin = std::fabs(_rFactor * _bandPass.rms());
            for (unsigned t = 0; t < nSamples; ++t) {
                int bin = -1;
                for (unsigned s = 0; s < nSubbands; ++s) {
                    I = stokesI -> spectrumData(t, s, 0);
                    for (unsigned c = 0; c < nChannels; ++c) {
                        float res = I[c] - medianDelta - _bandPass.intensityOfBin( ++bin );
                        if ( std::fabs(res) > margin ) {
                            //I[c] = _bandPass.intensityOfBin( bin ) + medianDelta;
                            I[c] = 0;
                        }
                    }
            }
/*
            for (unsigned t = 0; t < nSamples; ++t) {
                unsigned int bin = 0;
                for (unsigned s = 0; s < nSubbands; ++s) {
                    //sumI2 = 0.0;
                    sumI = 0.0;
                    I = stokesI -> spectrumData(t, s, 0);
                    for (unsigned c = 0; c < nChannels; ++c) {
                        copyI[c]=I[c];
                        sumI += I[c];
                        //sumI2 += pow(I[c],2);
                    }
                    // This gets you the median of each spectrum on the copy, data not affected
                    std::nth_element(copyI.begin(), copyI.begin()+copyI.size()/2, copyI.end());
                    subbandMedian[s]=(float)*(copyI.begin()+copyI.size()/2);

                    float res = subbandMedian[s] - _bandPass.intensityOfBin( s );
                    std::cout << "s=" << s << " res=" << res << " intensity=" << _bandPass.intensityOfBin( s ) << std::endl;
                    if ( res > margin ) {
                        for (unsigned int c = 0; c < nChannels; ++c) {
                            //I[c] -= res;
                            I[c] = 0;
                        }
                    }

                    // These get you the mean and rms which you probably don't need
                    //copySM[s] = subbandMedian[s];
                    //subbandMean[s] = sumI/nChannels;
                    //subbandRMS[s] = sqrt(sumI2/nChannels - pow(subbandMean[s],2));
                    //copySubbandRMS[s] = subbandRMS[s];
                }
*/

                // This is bad and needs to go. It gives you the median of the subband medians and RMSs

                //std::nth_element(copySM.begin(), copySM.begin()+copySM.size()/2, copySM.end());
                //medianOfMedians=*(copySM.begin()+copySM.size()/2);

                //std::nth_element(copySubbandRMS.begin(), copySubbandRMS.begin()+copySubbandRMS.size()/2, copySubbandRMS.end());
                //medianOfRMS=*(copySubbandRMS.begin()+copySubbandRMS.size()/2);


                // this is where the criteria are applied
                // if datum succeeds it is kept, if it fails, it is set to the bandpass value

/*
                for (unsigned s = 0; s < nSubbands; ++s) {
                    I = stokesI -> spectrumData(t, s, 0);
                    if (subbandMedian[s] > 2.0 * medianOfMedians){
                        //std::cout << "Clipping subband: " << s << std::endl;
                        //std::cout << subbandMedian[s] << " " << medianOfMedians << std::endl;
                        for (unsigned c = 0; c < nChannels; ++c) {
                            I[c]=medianOfMedians;
                        }
                    }
                    else{
                        for (unsigned c = 0; c < nChannels; ++c) {
                            if (fabs(I[c]-subbandMedian[s]) > 5. * medianOfRMS){
                                //std::cout << "Clipping subband, channel: " << std::endl;
                                //std::cout << s << " " << c << std::endl;
                                //std::cout << subbandMedian[s] << " " << subbandRMS[s] << std::endl;
                                //std::cout << medianOfMedians << " " << medianOfRMS << std::endl;
                                I[c]=subbandMedian[s];
                            }
                        }
                    }
                }
*/
            }
        }
    }

        
    
    
    
} // namespace lofar
} // namespace pelican
