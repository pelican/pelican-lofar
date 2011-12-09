#include "RFI_Clipper.h"
#include "SpectrumDataSet.h"
#include "WeightedSpectrumDataSet.h"
#include <QtCore/QFile>
#include <QtCore/QString>
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
  : AbstractModule( config ), _active(true), _crFactor(10.0),_srFactor(4.0), _current(0),
    _badSpectra(0)
{
    _current = 0;
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

    if( config.hasAttribute("channelRejectionRMS")  )
        _crFactor = config.getAttribute("channelRejectionRMS").toFloat();
    if( config.hasAttribute("spectrumRejectionRMS")  )
        _srFactor = config.getAttribute("spectrumRejectionRMS").toFloat();

    _maxHistory = config.getOption("History", "maximum", "10" ).toInt();
    _history.resize(_maxHistory);
    _historyNewSum.resize(_maxHistory);
    _historyMean.resize(_maxHistory);
    _historyRMS.resize(_maxHistory);
    _medianFromFile = _bandPass.median();
    _rmsFromFile = _bandPass.rms();
    _zeroDMing = 0;
    _num = 0; // _num is the number of points in the history
    _runningMedian = 0; // initialise the running median
    _integratedNewSum = 0;
    _integratedNewSumSq = 0;
    if( config.getOption("zeroDMing", "active" ) == "true" ) {
      _zeroDMing = 1;
    }
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
  
  /**                                                                                                                                       
   * @details The following if statement from RFI_Clipper.cpp tests the channels for abnormal intensity spikes.                             
   * @verbatim
   /**                                                                                                                                       
   * @details The following if statement from RFI_Clipper.cpp tests the channels for abnormal intensity spikes.                             
   * @verbatim                                                                                                                              
   if (I[index + c] - medianDelta > margin ) {                                                                                             
            I[index + c] = 0.0;                                                                                                           
            W[index +c] = 0.0;                                                                                                            
            for(unsigned int pol = 1; pol < nPolarisations; ++pol ) {                                                                     
              long index = stokesAll->index(s, nSubbands,                                                                                 
                                          pol, nPolarisations, t, nChannels );                                                            
              I[index + c] = 0.0;                                                                                                         
              W[index +c] = 0.0;                                                                                                          
            }                                                                                                                             
          }                                                                                                                               
          else{                                                                                                                           
            // Subtract the current model from the data                                                                                   
            I[index+c] -= bandPass[bin] + _zeroDMing * medianDelta ;                                                                      
            // if the condition doesn't hold build up the statistical                                                                     
            // description.                                                                                                               
            spectrumSum += I[index+c];                                                                                                    
                                                                                                                                          
            // Use this for RMS calculation - This is the RMS in the reference frame of the input                                         
            spectrumSumSq += pow(I[index+c],2);                                                                                           
            // Scale the data by the RMS                                                                                                  
            I[index+c] /= modelRMS;                                                                                                       
            ++goodChannels;                                                                                                               
          }                                                                                                                               
                                                                                                                              
   * @endverbatim                                                                                                                           
   *                                                                                                                                        
   * The next statements check the entire spectra against the current model. Discarding 
   * bad data and keeping good data, using it to update  \
   * the bandpass model used by the clipper.                                                                                                   
   *                                                                                                                                        
   * Lastly the RFI Clipper computes the RFI stats for each blob so that they are available for use.                                        
   *                                                                                                                                        
   */

// RFI clipper to be used with Stokes-I out of Stokes Generator
//void RFI_Clipper::run(SpectrumDataSetStokes* stokesAll)
void RFI_Clipper::run( WeightedSpectrumDataSet* weightedStokes )
{
  if( _active ) {
    float blobRMS = 0.0f;
    float blobSum = 0.0f;
    SpectrumDataSetStokes* stokesAll =
      static_cast<SpectrumDataSetStokes*>(weightedStokes->dataSet());
    SpectrumDataSet<float>* weights = weightedStokes->weights();
    float* I;
    unsigned nSamples = stokesAll->nTimeBlocks();
    unsigned nSubbands = stokesAll->nSubbands();
    unsigned nChannels = stokesAll->nChannels();
    unsigned nPolarisations = stokesAll->nPolarisations();
    unsigned nBins = nChannels * nSubbands;
    unsigned goodSamples = 0;
    float modelRMS = _bandPass.rms();
    // This has all been tested..
    _map.reset( nBins );
    _map.setStart( _startFrequency );
    _map.setEnd( _endFrequency );
    _bandPass.reBin(_map);
    _copyI.resize(nBins);
    // -------------------------------------------------------------
    // Processing next chunk 
    for (unsigned t = 0; t < nSamples; ++t) {
      float margin = std::fabs(_crFactor * _bandPass.rms());
      const QVector<float>& bandPass = _bandPass.currentSet();
      // The following is the amount of tolerance to changes in the average value of the spectrum
      float spectrumRMStolerance = _srFactor * _bandPass.rms()/sqrt(nBins);
      int bin = -1;
      float spectrumSum = 0.0;
      float spectrumSumSq = 0.0;
      float newSum = 0.0;
      float goodChannels = 0.0;
      float modelLevel = _bandPass.median();
      I = stokesAll->data();
      float *W = weights->data();

      // create a copy of the data minus the model in order
      // to compute the median. The median is used as a single number
      // to characterise the offset of the data and the model.
      bin = -1;
      for (unsigned s = 0; s < nSubbands; ++s) {
        long index = stokesAll->index(s, nSubbands,
                                    0, nPolarisations,
                                    t, nChannels );
        for (unsigned c = 0; c < nChannels ; ++c) {
          ++bin;
          _copyI[bin]=I[index + c] - bandPass[bin];
        }
      }

      // Compute the median of the flattened, model subtracted spectrum
      std::nth_element(_copyI.begin(), _copyI.begin()+_copyI.size()/2, _copyI.end());
      float median = (float)*(_copyI.begin()+_copyI.size()/2);

      // reset bin
      bin = -1;

      // Perform first test: look for individual very bright
      // channels in Stokes-I compared to the model and clip accordingly

      for (unsigned s = 0; s < nSubbands; ++s) {
        long index = stokesAll->index(s, nSubbands,
                                    0, nPolarisations,
                                    t, nChannels );
        for (unsigned c = 0; c < nChannels; ++c) {
          ++bin;

          // If (StokesI_of_channel_c -
          // bandPass_value_for_channel_bin) is greater than the
          // chosen margin blank that channel, if not add it to the
          // population of used channels for diagnostic purposes and
          // monitoring
          
          if (I[index + c] - bandPass[bin] - median> margin ) {
            I[index + c] = 0.0;
            W[index +c] = 0.0;
            for(unsigned int pol = 1; pol < nPolarisations; ++pol ) {
              long index = stokesAll->index(s, nSubbands,
                                          pol, nPolarisations, t, nChannels );
              I[index + c] = 0.0;
              W[index +c] = 0.0;
            }
          }
          else{
            
            // Subtract the current model from the data 
            I[index+c] -= bandPass[bin]; //+ _zeroDMing * median ;
            // if the condition doesn't hold build up the statistical
            // description;
            spectrumSum += I[index+c];
            // Use this for spectrum RMS calculation - This is the RMS
            // in the reference frame of the input
            spectrumSumSq += pow(I[index+c],2);
            ++goodChannels;
          }
        }
      }
      
      spectrumSum /= goodChannels;

      // This is the RMS of the model subtracted data, in the
      // reference frame of the input
      float spectrumRMS = sqrt(spectrumSumSq/goodChannels - std::pow(spectrumSum,2));

      // If goodChannels is substantially lower than the total number,
      // the rms of the spectrum will also be lower, so it needs to be
      // scaled by a factor sqrt(nBins/goodChannels)

      spectrumRMS *= sqrt(nBins/goodChannels);

      // Perform second test: Take a look at whether the median of the
      // spectrum is close to the model. If it isn't, it is likely
      // that the current spectrum has jumped in level compared to the
      // model, so something isn't right. If it is, then use the
      // median value to update the model

      // At this stage modelLevel is the bandPass.median()
      // spectrumRMStolerance is the estimated varience of
      // spectrumSum. 
      
      // This comparison is based on the values of the incoming data, i.e. before any scaling 
      //      if (fabs(medianDelta) > spectrumRMStolerance) {

      // Re compute the median of the model subtracted data now that
      // we know the highest (by definition) values may have been
      // chopped off
      if (goodChannels != nSubbands * nChannels){
        std::nth_element(_copyI.begin(), _copyI.begin()+goodChannels/2, _copyI.begin()+goodChannels);
        median = (float)*(_copyI.begin()+goodChannels/2);
      }

      // medianDelta is the level of the incoming data
      float medianDelta = median + _bandPass.median();


      if (fabs(median) > spectrumRMStolerance) {
      //      if (fabs(spectrumSum) > spectrumRMStolerance) {
        if (_badSpectra == 0) {
          std::cout 
            << "-------- RFI_Clipper----- Spectrum Average: " << spectrumSum << std::endl
            << " Median DataLevel: " << medianDelta 
            << " Median ModelLevel: " << modelLevel << std::endl
            << " Difference:  " << median 
            << " Tolerance: " << spectrumRMStolerance  << std::endl
            << " Good Channels: " << goodChannels 
            << " History Size: " << _history.size() << std::endl 
            << " SpectrumRMS: " << spectrumRMS 
            << " ModelRMS: " <<  _bandPass.rms() << std::endl
            << " _num:" << _num 
            << std::endl << std::endl;
        }

        //  Count how many bad spectra in a row. If number exceeds
        //  history, then reset model to parameters from bandpass file
        
        _badSpectra ++;
        if (_badSpectra == _history.size()){
          std::cout << "------ RFI_Clipper ----- Accepted a jump in the bandpass model to: " 
                    << medianDelta << " " << spectrumRMS  << std::endl << std::endl;
          _bandPass.setMedian(medianDelta);
          _bandPass.setRMS(spectrumRMS); // RMS in incoming reference frame
          _badSpectra = 0;
          // reset _num for the history calculations
          _num = 0 ;
          _current = 0;
        }

        // Clip entire spectrum
        for (unsigned s = 0; s < nSubbands; ++s) {
          long index = stokesAll->index(s, nSubbands,
                                      0, nPolarisations,
                                      t, nChannels );
          for (unsigned c = 0; c < nChannels; ++c) {
            I[index + c] = 0.0;
            W[index + c] = 0.0;
            for(unsigned int pol = 1; pol < nPolarisations; ++pol ) {
              long index = stokesAll->index(s, nSubbands,
                                          pol, nPolarisations, t, nChannels );
              I[index + c] = 0.0;
              W[index +c] = 0.0;
            }
          }
        }

      }
      else {
        // First take care of zero DM-ing, i.e. subtract the mean from
        // the data. Problem is, the data have been scaled by the
        // modelRMS, so spectrumSum needs to be scaled too, and a new
        // sum is computed
        for (unsigned s = 0; s < nSubbands; ++s) {
          long index = stokesAll->index(s, nSubbands,
                                        0, nPolarisations,
                                        t, nChannels );
          for (unsigned c = 0; c < nChannels; ++c) {
            // if the channel hasn't been clipped already, remove the spectrum average
            if (W[index+c] != 0.0){
              if (_zeroDMing == 1){
                I[index+c] -= spectrumSum;//spectrumRMS;//modelRMS;
              }
              // Scale the data by the RMS (this is potentially the last
              // place the data gets fiddled with, so must be done here)
              I[index+c] /= spectrumRMS;//spectrumRMS;//modelRMS;
              newSum += I[index+c];
            }
          }
        }
        // newSum contains the scaled and integrated spectrum, as a diagnostic
        // Yey! This spectrum has made it out of the clipper so consider it in the noise statistics
        _badSpectra = 0;
        ++goodSamples;
        blobSum += spectrumSum;
        blobRMS += spectrumRMS;
        // update historical data to the median value of the
        // current spectrum, since it has passed all the tests
        // and is good for comparison to the next spectrum

        // We will compute the running medianDelta, to keep track of
        // where the incoming data level is

        // We will also store a history of the integrated value of the
        // spectrum in order to compute its RMS, which is useful for
        // dedispersion.

        // medianDelta is in the reference frame of the incoming data so ok!
        // Store the median value
        _history[_current] = medianDelta;
        _historyNewSum[_current] = newSum;
        // update the history index (ring buffer) 
        _current = ++_current%_maxHistory;
        
        // if the buffer isn't full, update the average properly
        if (_num != _maxHistory ) {
          //          _runningMedian = (_runningMedian * (float) _num + median)/(float) (_num+1);
          _runningMedian = (_runningMedian * (float) _num + medianDelta)/(float) (_num+1);
          // store the integral of _historyNewSum and _historyNewSum^2 from the buffer
          _integratedNewSum += newSum;
          _integratedNewSumSq += pow(newSum,2);
          ++_num;
        }
        // Now the whole buffer is full. So I want to add the new
        // value and remove the first value from the running median
        else {
          _runningMedian = (_runningMedian * (float) _num - _history[_current] + medianDelta) / (float) _num;
        // store the integral of _historyNewSum and _historyNewSum^2 from the buffer
          _integratedNewSum += newSum - _historyNewSum[_current];
          _integratedNewSumSq += pow(newSum,2) - pow(_historyNewSum[_current],2);
        }
        //        Update the model to the current running median
        _bandPass.setMedian(_runningMedian);
      }
    }
    
    // Now the chunk has finished. All that remains is to pass on the stats of the chunk
    // 1. update the model RMS first
    if (goodSamples !=0){
      blobRMS /= goodSamples;
      _bandPass.setRMS(blobRMS);
    }
    // 2. Use the history of NewSum to compute a running blobSum and
    // blobRMS; blobSum is the value of the running mean of newSum, in
    // the output reference frame
    // Re-use the variable blobRMS to send out the integrated RMS value
    blobSum = _integratedNewSum / _num;
    blobRMS = sqrt( _integratedNewSumSq / _num - pow(blobSum,2));

    // This noise is the approximate value for a non-central
    // chi-squared distribution with 2N degrees of freedom, where N is
    // the number of channels
    // It only works if the data have been scaled to RMS=1
    if (_zeroDMing == 1){
      blobRMS = 2.0*sqrt(nChannels * nSubbands);
    }
    if (goodSamples !=0){
    weightedStokes->setRMS( blobRMS ); 
    weightedStokes->setMean( blobSum);
    }
    else {
      weightedStokes->setRMS( 1e+6 ); 
      weightedStokes->setMean( 0.0);
    }      
  }
}
} // namespace lofar
} // namespace pelican
