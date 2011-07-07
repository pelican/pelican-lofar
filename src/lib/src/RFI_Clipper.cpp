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
    : AbstractModule( config ), _active(true), _rFactor(3.0), _current(0)
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

    if( config.hasAttribute("rejectionFactor")  )
        _rFactor = config.getAttribute("rejectionFactor").toFloat();
    _maxHistory = config.getOption("History", "maximum", "10" ).toInt();
    _history.resize(_maxHistory);
    _num = 0; // _num is the number of points in the history
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

/*
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
v        float median = (float)*(_copyI.begin()+_copyI.size()/2);
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
*/

// RFI clipper to be used with Stokes-I out of Stokes Generator
//void RFI_Clipper::run(SpectrumDataSetStokes* stokesI)
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
    
    _copyI.resize(nBins);
    for (unsigned t = 0; t < nSamples; ++t) {
      float margin = std::fabs(_rFactor * _bandPass.rms());
      //float doubleMargin = margin * 2.0;
      const QVector<float>& bandPass = _bandPass.currentSet();
      // The following is the amount of tolerance to changes in the average value of the spectrum
      float spectrumRMStolerance = 5.0 * _bandPass.rms()/sqrt(nBins);
      int bin = -1;
      float spectrumSum = 0.0;
      float spectrumSumSq = 0.0;
      float goodChannels = 0.0;
      float modelLevel = _bandPass.median();
      I = stokesI->data();
      float *W = weights->data();
      
      // create an ordered copy of the data in order to compute the median
      for (unsigned s = 0; s < nSubbands; ++s) {
        long index = stokesI->index(s, nSubbands, 
                                    0, nPolarisations,
                                    t, nChannels );
        for (unsigned c = 0; c < nChannels; ++c) {
          _copyI[++bin]=I[index+c];
        }
      }
      std::nth_element(_copyI.begin(), _copyI.begin()+_copyI.size()/2, _copyI.end());
      float median = (float)*(_copyI.begin()+_copyI.size()/2);
      
      
      // medianDelta is the DC offset between the current spectrum and the model
      float medianDelta = median - modelLevel;
      
      
      // Perform first test: look for individual very bright
      // channels in Stokes-I compared to the model
      bin = -1;
      for (unsigned s = 0; s < nSubbands; ++s) {
        //                int bin = (s * nChannels) - 1;
        long index = stokesI->index(s, nSubbands, 
                                    0, nPolarisations,
                                    t, nChannels ); 
        for (unsigned c = 0; c < nChannels; ++c) {
          ++bin;
          
          // If the condition
          // StokesI_of_channel_c - bandPass_value_for_channel_bin
          // is greater than the chosen margin 
          // holds, blank that channel, if
          // not add it to the population of used channels
          //          std::cout << "Data: " << I[index+c] << " Model: " << _bandPass.intensityOfBin( bin ) << std::endl; 
          I[index+c] -= bandPass[bin];
          if (I[index + c] > margin ) {
            I[index + c] = 0.0;
            W[index +c] = 0.0;
            for(unsigned int pol = 1; pol < nPolarisations; ++pol ) {
              long index = stokesI->index(s, nSubbands, 
                                          pol, nPolarisations, t, nChannels ); 
              I[index + c] = 0.0;
              W[index +c] = 0.0;
            }
          }
          else{
            // if the condition doesn't hold, then subtract the
            // model from the data and build up the statistical
            // description; this should flatten the spectrum and
            // bring its average very close to 0
            spectrumSum += I[index+c];
            //                  spectrumSumSq += pow(I[index+c],2);
            ++goodChannels;
          }
        }
      }
      spectrumSum /= goodChannels;

      // This is the RMS of the residual, current data 
      //            float spectrumRMS = sqrt(spectrumSumSq/goodChannels - std::pow(spectrumSum,2));
      
      // Perform second test: the best estimate of the bandpass
      // has been removed from the data. SpectrumSum, which is
      // now the average, should therefore be close to zero. If
      // it isn't, it means that the data are far from the best
      // estimate of the bandpass, therefore the whole spectrum
      // should be flagged.
      
      // At this stage modelLevel is the bandPass.median()
      // spectrumRMStolerance is the estimated varience of
      // spectrumSum. 
      
      //if (fabs(spectrumSum) > spectrumRMStolerance) {
        //			if (fabs(spectrumSum) > spectrumRMStolerance) {
        if (fabs(medianDelta) > spectrumRMStolerance) {
          std::cout 
          << " SpectrumSum:" << spectrumSum 
                    << " Tolerance:" << spectrumRMStolerance 
          << " ModelLevel:" << modelLevel 
          << " Spectrum median:" << median 
          << " Good Channels:" << goodChannels 
          << " History Size:" << _history.size() 
          << " medianDelta:" << medianDelta 
          << " _num:" << _num 
          << std::endl;
        
        for (unsigned s = 0; s < nSubbands; ++s) {
          long index = stokesI->index(s, nSubbands, 
                                      0, nPolarisations,
                                      t, nChannels ); 
          for (unsigned c = 0; c < nChannels; ++c) {
            I[index + c] = 0.0;
            W[index + c] = 0.0;
            for(int pol = 1; pol < nPolarisations; ++pol ) {
              long index = stokesI->index(s, nSubbands, 
                                          pol, nPolarisations, t, nChannels ); 
              I[index + c] = 0.0;
              W[index +c] = 0.0;
            }
          }
        }
        
      }
      else {
        // update historical data to the median value of the
        // current spectrum, since it has passed all the tests
        // and is good for comparison to the next spectrum
        
        if (_num != _maxHistory ) ++_num;
        _history[_current] = median;
        _current = ++_current%_maxHistory;
        float baselineLevel = 0.0;
        for( int i=0; i< _num; ++i ) {
          baselineLevel += _history[i]; 
        }
        baselineLevel /= (float) _num;
        /*        std::cout << "Baseline: " << baselineLevel 
                  << " _num:" << _num 
                  << " current:" << _current
                  << " median:" << median 
                  << " spectrumSum:" << spectrumSum
                  << std::endl;*/
        _bandPass.setMedian(baselineLevel);
        //        _bandPass.setRMS(spectrumRMS);
        //        modelRMS = spectrumRMS;
        //        modelLevel = baselineLevel;
      }
      //}
    }
  }
}
  
} // namespace lofar
} // namespace pelican
