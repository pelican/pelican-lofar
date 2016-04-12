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
#include "omp.h"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
//#include "openmp.h"
namespace pelican {
namespace ampp {

/**
 *@details RFI_Clipper
 */
RFI_Clipper::RFI_Clipper( const ConfigNode& config )
  : AbstractModule( config ), _active(true), _crFactor(10.0),_srFactor(4.0), _current(0),
    _badSpectra(0), _fractionBadChannels(0)
{
    _current = 0;
    if( config.hasAttribute("active") &&
            config.getAttribute("active").toLower() == QString("false") ) {
        _active = false;
    }
    // read in any fixed file data
    QString file = config.getOption("BandPassData", "file", "");
    if( file != "" && _active ) {
        try {
            file=config.searchFile(file);
        }
        catch( QString e ) {
            throw(QString("RFI_Clipper: " + e ));
        }

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
    // Set the capacity of my running average circular buffers
    _meanBuffer.set_capacity(_maxHistory);
    _rmsBuffer.set_capacity(_maxHistory);
    _num = 0; // keep track of position in buffer for practical
	      // purposes
    _medianFromFile = _bandPass.median();
    _rmsFromFile = _bandPass.rms();
    _zeroDMing = 0;
    if( config.getOption("zeroDMing", "active" ) == "true" ) {
      _zeroDMing = 1;
    }
    if( config.getOption("computeRMSfromMean", "active", "true" ) == "true" ) {
      _useMeanOverRMS = 1;
    }
    _startFrequency = 0.0;
    _endFrequency = 0.0;
    if( config.getOption("Band", "matching" ) == "true" ) {
        _startFrequency = _bandPass.startFrequency();
        _endFrequency = _bandPass.endFrequency();
    }
    /*
    else {
        if( _active ) {
            if( config.getOption("Band", "startFrequency" ) != "" ) {
                _startFrequency = config.getOption("Band","startFrequency" ).toFloat();
            }

            if( config.getOption("Band", "endFrequency") != "" ) {
                _endFrequency = config.getOption("Band","endFrequency" ).toFloat();
            }

            if ((0.0 == _startFrequency) || (0.0 == _endFrequency))
            {
                // This is ALFABURST pipeline
                // calculate _startFrequency from LO frequency and number of channels used
                getLOFreqFromRedis();
                //TODO: do this properly, based on number of channels, which spectral
                //quarter, channel bandwidth, etc.
                _startFrequency = _LOFreq - (448.0 / 4);
                _endFrequency = _startFrequency - (448.0 / 8) + 0.109375;
            }
        }
    }
    */
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

   * @endverbatim
   *
   * The next statements check the entire spectra against the current model. Discarding
   * bad data and keeping good data, using it to update  \
   * the bandpass model used by the clipper.
   *
   * Lastly the RFI Clipper computes the RFI stats for each blob so that they are available for use.
   *
   */

  static inline void clipSample( SpectrumDataSetStokes* stokesAll, float* W, unsigned t, std::vector<float> lastGoodSpectrum ) {

    float* I = stokesAll->data();
    unsigned nSubbands = stokesAll->nSubbands();
    unsigned nPolarisations= stokesAll->nPolarisations();
    unsigned nChannels= stokesAll->nChannels();
    // Clip entire spectrum
    //#pragma omp parallel for num_threads(4)
    for (unsigned s = 0; s < nSubbands; ++s) {
      // The following is for clipping the polarization
      for(unsigned int pol = 0; pol < nPolarisations; ++pol ) {
        long index = stokesAll->index(s, nSubbands,
                pol, nPolarisations,
                t, nChannels );
        for (unsigned c = 0; c < nChannels; ++c) {
	  //            I[index + c] = 0.0;
	  W[index + c] = 1.0;
          //  W[index + c] = 0.0;
	  I[index + c] = lastGoodSpectrum[s*nChannels + c];
        }
      }
    }
}

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
    float* I = stokesAll->data();
    float *W = weights->data();
    unsigned nSamples = stokesAll->nTimeBlocks();
    unsigned nSubbands = stokesAll->nSubbands();
    unsigned nChannels = stokesAll->nChannels();
    unsigned nPolarisations = stokesAll->nPolarisations();
    unsigned nBins = nChannels * nSubbands;
    unsigned goodSamples = 0;
    float k = 4; // degrees of freedom
    float meanMinusMinimum = k / sqrt(2.0*k); 
    float spectrumRMS;
    float dataModel;

    if (_lastGoodSpectrum.size() != nBins) 
      {
	_lastGoodSpectrum.resize(nBins,0.0);
	_remainingZeros = nBins;
	std::cout << "RFI_Clipper: resizing _lastGoodSpectrum" << std::endl;
      }
    
    //float modelRMS = _bandPass.rms();
    // This has all been tested..
    _map.reset( nBins );
    _map.setStart( _startFrequency );
    _map.setEnd( _endFrequency );
    _bandPass.reBin(_map);
    // -------------------------------------------------------------
    // Processing next chunk 
    for (unsigned t = 0; t < nSamples; ++t) {
      //      std::cout << "Next spectrum " << _num << std::endl;

      const QVector<float>& bandPass = _bandPass.currentSet();
      float spectrumSum = 0.0;
      float spectrumSumSq = 0.0;
      // Split the spectrum into 8 bands for the purpose of matching
      // the spectrum to the model
      std::vector<float> goodChannels(8,0.0);
      unsigned channelsPerBand = nBins / 8;
      // find the minima of I in each band, and the minima of the
      // model and compare
      std::vector<float> miniData(8,1e6);
      std::vector<float> miniModel(8,1e6);
      std::vector<float> dataMinusModel(8);
      std::vector<float> bandSigma(8);
      std::vector<float> bandMean(8,0.0);
      std::vector<float> bandMeanSquare(8,0.0);

      // Find the data minima and model minima in each band
      // Let us also estimate sigma in each band
      
      // Try this over an adapting stage, lasting as long as the running average buffers
      if (!_rmsBuffer.full()){
	for (unsigned s = 0; s < nSubbands; ++s) {
	  long index = stokesAll->index(s, nSubbands,
					0, nPolarisations,
					t, nChannels );
	  for (unsigned c = 0; c < nChannels ; ++c) {
	    int binLocal = s*nChannels +c; 
	    unsigned band = (int)binLocal / channelsPerBand;
	    
	    //	  std::cout << "----------------" << std::endl; std::
	    //	  cout << band << " " << binLocal << " " << I[index+c]
	    //	  << std::endl;
	    
	    if (I[index+c] < miniData[band]) miniData[band] = I[index+c];
	    if (bandPass[binLocal] < miniModel[band]) miniModel[band] = bandPass[binLocal];
	    bandMean[band] += I[index+c];
	    bandMeanSquare[band] += (I[index+c]*I[index+c]);
	  }
	}
	
	// Now find the distances between data and model and the RMSs in
	// each band
	
	for (unsigned b = 0; b < 8; ++b){
	  dataMinusModel[b] = miniData[b] - miniModel[b];
	  bandSigma[b] = sqrt(bandMeanSquare[b]/channelsPerBand - std::pow(bandMean[b]/channelsPerBand,2));
	  //std::cout << bandSigma[b] << " " << bandMean[b] << " " <<
	  //bandMeanSquare[b] << std::endl;
	}
	// Assume the minimum bandSigma to be the best estimate of this
	// spectrum RMS
	spectrumRMS = *std::min_element(bandSigma.begin(), bandSigma.end());

	// Take the median of dataMinusModel to determine the distance
	// from the model
	std::nth_element(dataMinusModel.begin(), dataMinusModel.begin()+dataMinusModel.size()/2, dataMinusModel.end());
	dataModel = (float)*(dataMinusModel.begin()+dataMinusModel.size()/2);
	//      std::cout << "data minus model " << dataModel << " " <<
	// std::endl; since we have used the minima to determine this
	// distance, we assume that dataModel is actually k/sqrt(k)
	// sigma away from the real value, where k is the number of the
	// degrees of freedom of the chi-squared distribution of the
	// incoming data. For no integration, k will be 4 (2 powers per
	// poln)
	
	//      std::cout << "dataModel " << dataModel << std::endl;
	
	// Let us now build up a running average of spectrumRMS values
	// (_maxHistory of them)
	
	// if the buffer is not full, compute the new rmsRunAve like this
	// if (!_rmsBuffer.full()) {
	_rmsBuffer.push_back(spectrumRMS);
	_rmsRunAve = std::accumulate(_rmsBuffer.begin(), _rmsBuffer.end(), 0.0)/_rmsBuffer.size();
	
	// The very last time this is done, store a reference value of
	// the mean over the rms; this works as I have just added on the
	// last value two lines above
	if (_rmsBuffer.full()) _meanOverRMS = _meanRunAve / _rmsRunAve;
	
	// and update the model
	dataModel = dataModel + meanMinusMinimum * _rmsRunAve;

			
      }
      else {
	// just update the running average with the current last value,
	// and take the oldest off the end. Then add the new
	// value onto the buffer. The idea here is that the new
	// value is not used for the current spectrum, but rather
	// for the one after. The current spectrum is evaluated
	// based on the rms values in the buffer up to that
	// point.

	_rmsRunAve -= (_rmsBuffer.front()/_rmsBuffer.size());
	_rmsRunAve += (_rmsBuffer.back()/_rmsBuffer.capacity());

	// In extreme RFI cases, the measured RMS may start growing
	// due to particular RFI signals. The mean over rms ratio
	// should remain approximately constant over the course of the
	// observation. Use this as a safety check, after the mean has
	// been reasonably determined, to set the RMS to a more
	// reliable value :
	if  (_useMeanOverRMS)
	  //recover the rms running average
	  spectrumRMS = std::abs(_meanRunAve / _meanOverRMS);

	_rmsBuffer.push_back(spectrumRMS);
	// use the mean running average as a model of the mean of the
	// data; remember that the running average is updated with a
	// mean after channel clipping, below.
	dataModel = _meanRunAve;
      }
      
      // now use this rms to define a margin of tolerance for bright
      // channels
      float margin = _crFactor * _rmsRunAve;
      
      // Now loop around all the channels: if you find a channel where
      // (I - bandpass) - datamodel > margin, then replace it 
      for (unsigned s = 0; s < nSubbands; ++s) {
	long index = stokesAll->index(s, nSubbands,
				      0, nPolarisations,
				      t, nChannels );
	for (unsigned c = 0; c < nChannels; ++c) {
	  int binLocal = s*nChannels +c;
	  if (I[index+c] - dataModel - bandPass[binLocal] > margin) {
	    // clipping this channel to values from the last good
	    // spectrum 
	    /*std::cout << "clipping channel " << I[index+c]
		      << " " << dataModel << " " <<  bandPass[binLocal] 
		      << " " << margin << std::endl;*/
	    //The following is for polarization
	    for(unsigned int pol = 0; pol < nPolarisations; ++pol ) {
	      long index = stokesAll->index(s, nSubbands,
					    pol, nPolarisations, t, nChannels );
	      I[index + c] = _lastGoodSpectrum[binLocal];
	      W[index +c] = 1.0;
	      //	      W[index +c] = 0.0;
	    }
	  }
	  else{
	    unsigned band = (int)binLocal / channelsPerBand;
	    ++goodChannels[band];
	    spectrumSum += I[index+c];
	  }
	}
      }
      // So now we have the mean of the incoming data, in a reliable
      // form after channel clipping
      unsigned totalGoodChannels=std::accumulate(goodChannels.begin(), goodChannels.end(), 0);
      _fractionBadChannels += (float)(nBins - totalGoodChannels)/nBins;
      spectrumSum /= totalGoodChannels;

      // Check if more than 20% of the channels in each band were
      // bad. If so in more than half of the bands, 4 in this case,
      // keep record. Also, if one band is completely gone, or less
      // than 80% of the total survive, keep record.
      

      unsigned badBands = 0; 
      for (unsigned b = 0; b < 8; ++b){
	if (goodChannels[b] < 0.8 * channelsPerBand) {
	  ++badBands;
	}
	if (goodChannels[b] == 0) {
	  badBands += 4;
	}
      }
      if (totalGoodChannels < 0.8 * nBins) badBands +=4;

      //      std::cout << "Current model, mean and rms: " <<
      //      _bandPass.mean() << " " << _bandPass.rms() << std::endl;
      //      std::cout << "Current data, mean and rms: " <<
      //      spectrumSum << " " << spectrumRMS << std::endl;
      //      std::cout << "new dataModel: " << dataModel <<
      //      std::endl;

      //      std::cout << "good channels " << goodChannels <<
      //      std::endl;
      


      // Let us now build up the running average of spectrumSum values
      // (_maxHistory of them) if the buffer is not full, compute the
      // new meanRunAve like this


      if (!_meanBuffer.full()) {
	_meanBuffer.push_back(spectrumSum);
	_meanRunAve = std::accumulate(_meanBuffer.begin(), _meanBuffer.end(), 0.0)/_meanBuffer.size();
      }
      else {
	//   just update the running average with the new value, and
	//   take the oldest off the end, using the same principle as
	//   with the rms buffer, i.e. do not use the current
	//   measurement for the current spectrum.

	// Note there is a tiny descrepance at the point when the
	// buffer is first full

	//	std::cout << "History buffer now full " << _num << std::endl;
	_meanRunAve -= _meanBuffer.front()/_meanBuffer.size();
      	_meanRunAve += _meanBuffer.back()/_meanBuffer.size();
      	_meanBuffer.push_back(spectrumSum);
      }

      // Now we can check if this spectrum has an abnormally high mean
      // compared to the running average
      
      // Let us define the tolerance first, and remember, we are
      // summing across nBins, hence sqrt(nBins), strictly only valid
      // for Gaussian stats
      float spectrumRMStolerance = _srFactor * _bandPass.rms()/sqrt(nBins); 

      //Now check, if spectrumSum - model > tolerance, declare this
      //time sample useless, replace its data and take care of the
      //running averages, also cut the first 1000 spectra, also cut
      //spectra where badBands >= 4, see above

      if (_meanBuffer.size() < 1000) {
	// clip the sample, but continue to build the stats; this
	// helps the stats converge
	clipSample( stokesAll, W, t, _lastGoodSpectrum );
      }
      else if (spectrumSum - _meanRunAve > spectrumRMStolerance || badBands >= 4) {

	// we need to remove this entire spectrum
	clipSample( stokesAll, W, t, _lastGoodSpectrum );
	// keep a record of bad spectra
	++_badSpectra;

	// now remove the last samples from the running average
	// buffers and replace them with the last good values.
	// 
	_meanBuffer.pop_back();	
	_rmsBuffer.pop_back();
	//	_meanBuffer.push_back(_lastGoodMean);
	//	_rmsBuffer.push_back(_lastGoodRMS);
	_meanBuffer.push_back(_meanBuffer.back());
	_rmsBuffer.push_back(_rmsBuffer.back());
      }
      // else keep a copy of the original spectrum, as it is good

      else {
	spectrumSum = 0.0;
	spectrumSumSq = 0.0;
	//	if (!_lastGoodSpectrum.full())
	// if _lastGoodSpectrum is full, _remainingZeros will be 0
	if (_remainingZeros != 0) 
	  {
	    for (unsigned s = 0; s < nSubbands; ++s) {
	      long index = stokesAll->index(s, nSubbands,
					    0, nPolarisations,
					    t, nChannels );
	      for (unsigned c = 0; c < nChannels; ++c) {
		int binLocal = s*nChannels +c;
		if (_lastGoodSpectrum[binLocal] == 0.0 && I[index+c] != 0.0 ){
		  _lastGoodSpectrum[binLocal] = I[index+c];
		  --_remainingZeros;
		  std::cout << "remaining channels to fill in good spectrum: " 
			    << _remainingZeros << " " << _num  
			    << " " << totalGoodChannels << std::endl;
		  spectrumSum += _lastGoodSpectrum[binLocal];
		  spectrumSumSq += _lastGoodSpectrum[binLocal] * 
		    _lastGoodSpectrum[binLocal];
		}
	      }
	    }
	    // and keep the mean and rms values as computed
	    _lastGoodMean = spectrumSum / nBins;
	    _lastGoodRMS = sqrt(spectrumSumSq/nBins - std::pow(_lastGoodMean,2));
	  }
      }
      
      
      // Now we have a valid spectrum, either the original or
      // replaced; this spectrum is good, so let us do the final
      // bits of post processing reset the spectrumSum, and SumSq,
      // and flatten subtract the bandpass from the data.
      // 
      spectrumSum = 0.0;
      spectrumSumSq = 0.0;
      for (unsigned s = 0; s < nSubbands; ++s) {
	long index = stokesAll->index(s, nSubbands,
				      0, nPolarisations,
				      t, nChannels );
	for (unsigned c = 0; c < nChannels; ++c) {
	  int binLocal = s*nChannels +c;
	  //std::cout << "here1 " << bandPass[binLocal] << " " << dataModel << std::endl;
	  // flat bandpass with near zero mean
	  I[index+c] -= (bandPass[binLocal] + dataModel); 
	  //std::cout << "here2" << std::endl;
	  spectrumSum += I[index+c];
	  //std::cout << "here3" << std::endl;
	  spectrumSumSq += I[index+c]*I[index+c];
	}
      }

      // and normalize: bring to zero mean if zerodm is specified or
      // use the running mean if not
      spectrumSum /= nBins; // New meaning of these two variables
      spectrumRMS = sqrt(spectrumSumSq/nBins - std::pow(spectrumSum,2));
      //std::cout << "Values used for normalization" << std::endl;
      //std::cout << spectrumSum << " " << spectrumRMS << std::endl;

      // Avoid nastiness in those first spectra by avoiding divisions
      // by zero
      if (spectrumRMS == 0.0) spectrumRMS = 1.0;
      
      
      for (unsigned s = 0; s < nSubbands; ++s) {
	long index = stokesAll->index(s, nSubbands,
				      0, nPolarisations,
				      t, nChannels );
	for (unsigned c = 0; c < nChannels; ++c) {
	  if (_zeroDMing == 1)
	    {
	      I[index+c] -= _zeroDMing * spectrumSum;
	    }
	  else
	    {
	      I[index+c] -= _meanRunAve;
	    }
	  // it may be better to normalize by the running average RMS,
	  // given this is a sensitive operation. For example, an
	  // artificially low rms may scale things up
	  I[index+c] /= spectrumRMS;
	  //	  I[index+c] /= _rmsRunAve;
	  // make sure this division is not introducing signals that
	  // you would have clipped
	  if (I[index+c] > _crFactor) I[index+c] = 0.0; 
	}
      }
      // The bandpass is flat and the spectrum clean and normalized,
      // so move to the next spectrum and write out some stats:
      unsigned reportStatsEvery = 10 * _maxHistory;
      if (_num == 0) {
	// calculate fractions
	float fractionBadSpectra = 100.0 * (float)_badSpectra / (float)reportStatsEvery; 
	float fractionBadChannels = 100.0 * _fractionBadChannels / (float)reportStatsEvery ;

	// if the fraction of bad spectra becomes >99%, then empty the
	// circular buffers and go into learning mode again
	if (fractionBadSpectra > 99.0) {
	  _rmsBuffer.resize(0);
	  _meanBuffer.resize(0);
	  std::cout << "Lost track of the RFI model, retraining.";
	}
	
	std::cout << std::endl;
	//	std::cout << "# Mean RMS %badSpectra %badChannels : "
	//	<< std::endl;
	std::cout <<  "RFIstats: " << _meanRunAve << " " << _rmsRunAve << " " 
		  << fractionBadSpectra << " " << fractionBadChannels << std::endl;
	std::cout << std::endl;
	std::cout << nBins << std::endl;
	// Reset _bad
	_badSpectra = 0;
	_fractionBadChannels = 0.0;

      }    
      // and update the model
      _bandPass.setMean(_meanRunAve);
      _bandPass.setRMS(_rmsRunAve);
      ++_num;
      _num = _num % reportStatsEvery;
      
    }
    // set the stats of the chunk
    weightedStokes->setRMS( _rmsRunAve );
    weightedStokes->setMean( _meanRunAve);
  }
}
} // namespace ampp
} // namespace pelican
