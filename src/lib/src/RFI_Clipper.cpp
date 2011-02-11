#include "RFI_Clipper.h"
#include "SpectrumDataSet.h"

namespace pelican {
    
namespace lofar {
        
    
    /**
     *@details RFI_Clipper 
     */
    RFI_Clipper::RFI_Clipper( const ConfigNode& config )
        : AbstractModule( config ), _active(true)
    {
        if( config.hasAttribute("active") &&  config.getAttribute("active").toLower() == QString("false") ) {
            _active = false;
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
            std::vector<float> copyI(nChannels);
            std::vector<float> subbandMean(nSubbands);
            std::vector<float> subbandMedian(nSubbands);
            std::vector<float> copySM(nSubbands);
            std::vector<float> subbandRMS(nSubbands);
            std::vector<float> copySubbandRMS(nSubbands);
            float medianOfMedians, medianOfRMS, sumI, sumI2 ;

            for (unsigned t = 0; t < nSamples; ++t) {

                for (unsigned s = 0; s < nSubbands; ++s) {
                    sumI2 = 0.0;
                    sumI = 0.0;
                    I = stokesI -> spectrumData(t, s, 0);
                    for (unsigned c = 0; c < nChannels; ++c) {
                        copyI[c]=I[c];
                        sumI += I[c];
                        sumI2 += pow(I[c],2);
                    }
                    std::nth_element(copyI.begin(), copyI.begin()+copyI.size()/2, copyI.end());
                    subbandMedian[s]=(float)*(copyI.begin()+copyI.size()/2);

                    copySM[s] = subbandMedian[s];
                    subbandMean[s] = sumI/nChannels;
                    subbandRMS[s] = sqrt(sumI2/nChannels - pow(subbandMean[s],2));

                    copySubbandRMS[s] = subbandRMS[s];
                }

                std::nth_element(copySM.begin(), copySM.begin()+copySM.size()/2, copySM.end());
                medianOfMedians=*(copySM.begin()+copySM.size()/2);

                std::nth_element(copySubbandRMS.begin(), copySubbandRMS.begin()+copySubbandRMS.size()/2, copySubbandRMS.end());
                medianOfRMS=*(copySubbandRMS.begin()+copySubbandRMS.size()/2);

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
            }
        }
    }

        
    
    
    
} // namespace lofar
} // namespace pelican
