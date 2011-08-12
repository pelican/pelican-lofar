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
  : AbstractModule( config ), _active(true), _crFactor(10.0),_srFactor(4.0), _current(0), _currentChunk(0),
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
    _historyMean.resize(_maxHistory);
    _historyRMS.resize(_maxHistory);
    _medianFromFile = _bandPass.median();
    _rmsFromFile = _bandPass.rms();

    _num = 0; // _num is the number of points in the history
    _numChunks = 0; // _numChunks is the number of chunks in the history
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
        SpectrumDataSetStokes* stokesAll =
            static_cast<SpectrumDataSetStokes*>(weightedStokes->dataSet());
        SpectrumDataSet<float>* weights = weightedStokes->weights();
        float* I;
        unsigned nSamples = stokesAll->nTimeBlocks();
        unsigned nSubbands = stokesAll->nSubbands();
        unsigned nChannels = stokesAll->nChannels();
        unsigned nPolarisations = stokesAll->nPolarisations();
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
                I = stokesAll -> spectrumData(t, s, 0);
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
        I = stokesAll->data();
        float *W = weights->data();
        for (unsigned t = 0; t < nSamples; ++t) {
#pragma omp parallel for
            for (unsigned s = 0; s < nSubbands; ++s) {
                int bin = (s * nChannels) - 1;
                long index = stokesAll->index(s, nSubbands,
                        0, nPolarisations,
                        t, nChannels );

                //float *I = stokesAll -> spectrumData(t, s, 0);
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

    _map.reset( nBins );
    _map.setStart( _startFrequency );
    _map.setEnd( _endFrequency );
    _bandPass.reBin(_map);

    _copyI.resize(nBins);
    for (unsigned t = 0; t < nSamples; ++t) {
      float margin = std::fabs(_crFactor * _bandPass.rms());
      //float doubleMargin = margin * 2.0;
      const QVector<float>& bandPass = _bandPass.currentSet();
      // The following is the amount of tolerance to changes in the average value of the spectrum
      //      float spectrumRMStolerance = 5.0 * _bandPass.rms()/sqrt(nBins);
      float spectrumRMStolerance = _srFactor * _bandPass.rms()/sqrt(nBins);
      int bin = -1;
      float spectrumSum = 0.0;
      float spectrumSumSq = 0.0;
      float goodChannels = 0.0;
      float modelLevel = _bandPass.median();
      I = stokesAll->data();
      float *W = weights->data();

      // create an ordered copy of the data in order to compute the median
      // The median is used as a single number to characterise the level of each spectrum

      bin = -1;
      for (unsigned s = 0; s < nSubbands; ++s) {
        long index = stokesAll->index(s, nSubbands,
                                    0, nPolarisations,
                                    t, nChannels );
        for (unsigned c = 0; c < nChannels; ++c) {
          ++bin;
          _copyI[bin]=I[index+c];
        }
      }

      // Compute the median
      std::nth_element(_copyI.begin(), _copyI.begin()+_copyI.size()/2, _copyI.end());
      float median = (float)*(_copyI.begin()+_copyI.size()/2);

      // medianDelta is the DC offset between the current spectrum and the model
      float medianDelta = median - modelLevel;

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
            I[index+c] -= bandPass[bin] + medianDelta ;
            // if the condition doesn't hold build up the statistical
            // description;
            spectrumSum += I[index+c];

            // Use this for RMS calculation
            spectrumSumSq += pow(I[index+c],2);
            ++goodChannels;
          }
        }
      }

      spectrumSum /= goodChannels;

      // This is the RMS of the residual, current data
      float spectrumRMS = sqrt(spectrumSumSq/goodChannels - std::pow(spectrumSum,2));

      // Perform second test: Take a look at whether the median of the
      // spectrum is close to the model. If it isn't, it is likely
      // that the current spectrum has jumped in level compared to the
      // model, so something isn't right. If it is, then use the
      // median value to update the model

      // At this stage modelLevel is the bandPass.median()
      // spectrumRMStolerance is the estimated varience of
      // spectrumSum.

      if (fabs(medianDelta) > spectrumRMStolerance) {
        if (_badSpectra == 0) {
          std::cout
            << "-------- RFI_Clipper----- SpectrumSum:" << spectrumSum
            << " Tolerance:" << spectrumRMStolerance << std::endl
            << " ModelLevel:" << modelLevel
            << " Spectrum median:" << median << std::endl
            << " Good Channels:" << goodChannels
            << " History Size:" << _history.size()
            << " medianDelta:" << medianDelta
            << " _num:" << _num
            << std::endl;
        }

        //  Count how many bad spectra in a row. If number exceeds history, then reset model to parameters from bandpass file

        _badSpectra ++;
        if (_badSpectra == _history.size()){
          std::cout << "------ RFI_Clipper ----- REVERTING TO BANDPASS FILE MODEL" << std::endl;
          _bandPass.setMedian(_medianFromFile);
          _bandPass.setRMS(_rmsFromFile);
          _badSpectra = 0;
        }


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
        // Yey! This spectrum has made it out of the clipper so consider it in the noise statistics
        _badSpectra = 0;
        ++goodSamples;
        blobSum += spectrumSum;
        blobRMS += spectrumRMS;

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

        /*        if (_current == 1) std::cout
                             << "**** RFI_Clipper ---------> "
                             << " Baseline at: " << baselineLevel
                             << " Tolerance: " << 5.0 * spectrumRMS/sqrt(nBins)
                             << " Spectrum Mean: " << spectrumSum
                             << " Good Channels: " << goodChannels
                             << std::endl
                             << std::endl;
        */

        /*        std::cout << "Baseline: " << baselineLevel
                  << " _num:" << _num
                  << " current:" << _current
                  << " median:" << median
                  << " spectrumSum:" << spectrumSum
                  << std::endl;*/
        _bandPass.setMedian(baselineLevel);
        _bandPass.setRMS(spectrumRMS);
      }
    }

    // Pass a running average mean from XX chunks as the noise properties of this chunk
    // Where XX is the history size devided by the number of samples per chunk

    if (goodSamples !=0){
      blobRMS /= (goodSamples*sqrt(nBins));
      blobSum /= goodSamples;
    }

    if (_numChunks != _maxHistory / nSamples ) ++_numChunks;
    _historyMean[_currentChunk] = blobSum;
    _historyRMS[_currentChunk] = blobRMS;
    _currentChunk = ++_currentChunk%(_maxHistory / nSamples) ;
    float runningMean = 0.0;
    float runningRMS = 0.0;
    for( int i=0; i< _numChunks; ++i ) {
      runningMean += _historyMean[i];
      runningRMS += _historyRMS[i];
    }
    runningMean /= _numChunks;
    runningRMS /= _numChunks;

    weightedStokes->setRMS( runningRMS );
    weightedStokes->setMean( runningMean );
    /*
    if (_currentChunk == 1) std::cout
                              << "**** RFI_Clipper ---------> "
                              << " Running mean at : " << runningMean
                              << " Running RMS at : " << runningRMS
                              << " Chunks processed : " << _numChunks
                              << std::endl
                              << std::endl;
    */
  }
}
} // namespace lofar
} // namespace pelican
