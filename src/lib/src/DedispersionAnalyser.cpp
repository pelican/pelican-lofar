#include "DedispersionAnalyser.h"
#include "DedispersionSpectra.h"
#include "DedispersionDataAnalysis.h"
#include "SpectrumDataSet.h"
#include <QVector>


namespace pelican {

namespace ampp {


/**
 *@details DedispersionAnalyser 
 */
DedispersionAnalyser::DedispersionAnalyser( const ConfigNode& config )
    : AbstractModule( config )
{
    // Get configuration options                                                                                                                      
    //unsigned int nChannels = config.getOption("outputChannelsPerSubband", "value", "512").toUInt();                                                 
    _detectionThreshold = config.getOption("detectionThreshold", "in_sigma", "6.0").toFloat();
    _binPow2 = config.getOption("power2ForBinning", "value", "6").toUInt();
    _useStokesStats = config.getOption("useStokesStats", "0_or_1").toUInt();
}

DedispersionAnalyser::~DedispersionAnalyser()
{
}

int DedispersionAnalyser::analyse( DedispersionSpectra* data, 
                                    DedispersionDataAnalysis* result ) {

    result->reset(data);
    const QList<SpectrumDataSetStokes* >& d = data->inputDataBlobs(); 
    int nChannels = d[0]->nChannels();
    int nSubbands = d[0]->nSubbands();

    float rms = std::sqrt((float)nChannels*(float)nSubbands);
    result->setRMS(rms);

    std::cout << "---------------" << nChannels << " " << nSubbands << std::endl;
    // Calculate the mean
    //double mean = 0.0, stddev = 0.0;
    double total = 0.0;

    QVector<float> dataVector = data->data();
    /*
    int vals=dataVector.size();
    for( int j = 0; j < vals; ++j ) {
        total += dataVector[j];
    }
    mean = total/vals;  // Mean for entire array
    std::cout << "Mean: " << mean << std::endl;

    // Calculate standard deviation
    total = 0;
    for(int j=0; j < vals; ++j ) {
        total += pow(dataVector[j] - mean, 2);
    }
    stddev = sqrt(total / vals); // Stddev for entire dedispersion array
    std::cout << "Stddev: " << stddev << std::endl;
    // Subtract dm mean from all samples and apply threshold
    */
    int tdms = data->dmBins();
    int nsamp = data->timeSamples();

    // Check if the buffer is lost

    // Add a dummy event to get the timestamp of the first bin in the blob
    result->addEvent( 0, 0, 1, 0.0 );
    
    // Compute 2^_binPowerOf2
    unsigned int maxPow2 = pow(2,_binPow2);
    unsigned int numberOfwidestBins = nsamp/maxPow2;
    // Attempt at tidying up the stuff below
    // Define a vector of _binPowerOf2 vectors
    QVector < QVector <float> > binnedOutput;
    binnedOutput.resize(_binPow2 + 1);
    // resize in the other direction
    // 1 bin for highest index, 2 for next, 4, 8 etc.
    unsigned int currentPow2 = maxPow2;
    for (int n = _binPow2 ; n > -1 ; --n){
      binnedOutput[n].resize(maxPow2 / currentPow2);
      currentPow2 /= 2;
    }
    // 
    unsigned int numberOfWidestBins = nsamp / maxPow2;
    for(int dm_count = 0; dm_count < tdms; ++dm_count) {

      /*
      QVector<float> outputBin1;
      QVector<float> outputBin2;
      QVector<float> outputBin4;
      QVector<float> outputBin8;
      QVector<float> outputBin16;
      float outputBin32;

      outputBin1.resize(32);
      outputBin2.resize(16);
      outputBin4.resize(8);
      outputBin8.resize(4);
      outputBin16.resize(2);
      */
      //    outputBin8.resize(1);
      
      for (int i=0; i < numberOfWidestBins; ++i) {
        currentPow2 = maxPow2;
        // fill the bin1 Vector first and check
        for (int j=0; j<maxPow2; ++j){
          int index = i*32 + j;
          binnedOutput[0][j]= dataVector[dm_count*nsamp + index]; 
          float detection = _detectionThreshold * rms;// * sqrt((float)indexJ);
          if (binnedOutput[0][j] >= detection){
            result->addEvent( dm_count, index, 1, binnedOutput[0][j] );
          }
        }
        for (int n = 1 ; n < _binPow2 + 1; ++n){
          currentPow2 /= 2; // 
          for (int j = 0; j < currentPow2; ++j){
            int binFactor = maxPow2/currentPow2;
            float detection = _detectionThreshold * rms * sqrt((float)binFactor);
            int index = i*currentPow2 + binFactor * j;
            binnedOutput[n][j] = binnedOutput[n-1][2*j] + binnedOutput[n-1][2*j+1];
            if (binnedOutput[n][j] >= detection){
              result->addEvent( dm_count, index, binFactor, binnedOutput[n][j] );
            }
          }
        }
      }
    }
        /*
      for(int i=0; i < nsamp/32; ++i) {
        for(int j=0; j < 32; ++j) {
          int index = i*32 + j;
          outputBin1[j]= dataVector[dm_count*nsamp + index]; 
          if (outputBin1[j] >= _detectionThreshold * rms){
            result->addEvent( dm_count, index, 1, outputBin1[j] );
            //                std::cout << outputBin1[j] << std::endl;
          }
        }
        for(int j=0; j < 16; ++j) {
          int index = i*32 + 2*j;
          outputBin2[j]= outputBin1[2*j] + outputBin1[2*j+1];
          if (outputBin2[j] >= _detectionThreshold * rms * 1.4142){
            result->addEvent( dm_count, index, 2, outputBin2[j] );
            //                std::cout << "2 " <<outputBin2[j] << std::endl;
          }
        }
        for(int j=0; j < 8; ++j) {
          int index = i*32 + 4*j;
          outputBin4[j]= outputBin2[2*j] + outputBin2[2*j+1];
          if (outputBin4[j] >= _detectionThreshold * rms * 2){
            result->addEvent( dm_count, index, 4, outputBin4[j] );
            //                std::cout << "4 " <<outputBin4[j] << std::endl;
            //            result->addEvent( dm_count, index );
          }
        }
        for(int j=0; j < 4; ++j) {
          int index = i*32 + 8*j;
          outputBin8[j]= outputBin4[2*j] + outputBin4[2*j+1];
          if (outputBin8[j] >= _detectionThreshold * rms * 2.8284){
            result->addEvent( dm_count, index, 8, outputBin8[j] );
            //              std::cout <<"8 " << outputBin8[j] << std::endl;
            //            result->addEvent( dm_count, index );
          }
        }
        for(int j=0; j < 2; ++j) {
          int index = i*32 + 16*j;
          outputBin16[j]= outputBin8[2*j] + outputBin8[2*j+1];
          //          if (outputBin16[j] >= _detectionThreshold * rms * 4){
          if (outputBin16[j] >= _detectionThreshold * rms * 4.0){
            //            result->addEvent( dm_count, index );
            //              std::cout << "16 " <<outputBin16[j] << std::endl;
            result->addEvent( dm_count, index, 16, outputBin16[j] );
            
          }
        }
        int index = i*32;
        outputBin32 = outputBin16[0] + outputBin16[1];
        //        if (outputBin32 >= _detectionThreshold * rms * 5.657){
        if (outputBin32 >= _detectionThreshold * rms * 5.67){
          //          result->addEvent( dm_count, index );
          //              std::cout << "32 " << outputBin32 << std::endl;
          result->addEvent( dm_count, index, 32, outputBin32);
          
        }
      }
      }
        */

    /*
    for(int dm_count = 0; dm_count < tdms; ++dm_count) {
        for(int j=0; j < nsamp; ++j) {
            total = dataVector[dm_count*nsamp + j];
            if (total >= _detectionThreshold * rms){
                result->addEvent( dm_count, j );
            }
        }
    }
    */
    std::cout << "Found " << result->eventsFound() << " events" << std::endl;
    return result->eventsFound();
}


} // namespace ampp
} // namespace pelican
