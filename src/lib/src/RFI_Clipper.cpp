#include "RFI_Clipper.h"
#include "SpectrumDataSet.h"

namespace pelican {
    
namespace lofar {
        
    
    /**
     *@details RFI_Clipper 
     */
    RFI_Clipper::RFI_Clipper( const ConfigNode& config )
        : AbstractModule( config )
    {
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
        float* I;
        unsigned nSamples = stokesI->nTimeBlocks();
        unsigned nSubbands = stokesI->nSubbands();
        unsigned nChannels = stokesI->nChannels();
        std::vector<float> copyI(nChannels);
        std::vector<float> subbandMean(nSubbands);
        std::vector<float> subbandMedian(nSubbands);
        std::vector<float> copySM(nSubbands);
        std::vector<float> subbandRMS(nSubbands);
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
                subbandMedian[s]=*(copyI.begin()+copyI.size()/2);

                copySM[s] = subbandMedian[s];
                subbandMean[s] = sumI/nChannels;
                subbandRMS[s] = sqrt(sumI2/nChannels - pow(subbandMean[s],2));

            }

            std::nth_element(copySM.begin(), copySM.begin()+copySM.size()/2, copySM.end());
            medianOfMedians=*(copySM.begin()+copySM.size()/2);

            std::nth_element(subbandRMS.begin(), subbandRMS.begin()+subbandRMS.size()/2, subbandRMS.end());
            medianOfRMS=*(subbandRMS.begin()+subbandRMS.size()/2);
            
            for (unsigned s = 0; s < nSubbands; ++s) {
                I = stokesI -> spectrumData(t, s, 0);
                
                if (subbandMedian[s] > 2.0 * medianOfMedians){
                    //                    std::cout << s << std::endl;
                    for (unsigned c = 0; c < nChannels; ++c) {
                        I[c]=0.0;
                    }
                }
                else{
                    for (unsigned c = 0; c < nChannels; ++c) {
                        I[c] -= subbandMedian[s];
                        //                        if (fabs(I[c]) >= 5.0 * subbandRMS[s]){
                        if (fabs(I[c]) >= 5.0 * medianOfRMS){
                            //                            if (s == 9){
                                //                            std::cout << s << " " << c << std::endl;
                                //                            std::cout << fabs(I[c]) << " " << subbandRMS[s] << std::endl;}
                            I[c]=0.0;
                        }
                    }
                }
            }
        }
    }
        
        
    
    
    
} // namespace lofar
} // namespace pelican
