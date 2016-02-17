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
#pragma omp parallel for num_threads(4)
    for (unsigned s = 0; s < nSubbands; ++s) {
      // The following is for clipping the polarization
      for(unsigned int pol = 0; pol < nPolarisations; ++pol ) {
        long index = stokesAll->index(s, nSubbands,
                pol, nPolarisations,
                t, nChannels );
        for (unsigned c = 0; c < nChannels; ++c) {
	  //            I[index + c] = 0.0;
            W[index + c] = 1.0;
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

    if (_lastGoodSpectrum.size() != nBins) _lastGoodSpectrum.resize(nBins);

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
      // The following is the amount of tolerance to changes in the average value of the spectrum
      float spectrumSum = 0.0;
      float spectrumSumSq = 0.0;
      //      float goodChannels = 0.0;
      std::vector<float> goodChannels(8,0.0);
      // Split the spectrum into 8 bands for the purpose of matching the spectrum to the model
      unsigned channelsPerBand = nBins / 8;
      // find the minima of I in each band, and the minima of the model and compare
      std::vector<float> miniData(8,1e6);
      std::vector<float> miniModel(8,1e6);
      std::vector<float> dataMinusModel(8);
      std::vector<float> bandSigma(8);
      std::vector<float> bandMean(8,0.0);
      std::vector<float> bandMeanSquare(8,0.0);

      // Find the data minima and model minima in each band
      // Let us also estimate sigma in each band
      for (unsigned s = 0; s < nSubbands; ++s) {
        long index = stokesAll->index(s, nSubbands,
				      0, nPolarisations,
				      t, nChannels );
        for (unsigned c = 0; c < nChannels ; ++c) {
          int binLocal = s*nChannels +c; 
	  unsigned band = (int)binLocal / channelsPerBand;

	  //	  std::cout << "----------------" << std::endl;
	  //	  std:: cout << band << " " << binLocal << " " << I[index+c] << std::endl;

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
	//std::cout << bandSigma[b] << " " << bandMean[b] << " " << bandMeanSquare[b] << std::endl;
      }
      // Assume the minimum bandSigma to be the best estimate of this spectrum RMS
      float spectrumRMS = *std::min_element(bandSigma.begin(), bandSigma.end());

      // Take the median of dataMinusModel to determine the distance from the model
      std::nth_element(dataMinusModel.begin(), dataMinusModel.begin()+dataMinusModel.size()/2, dataMinusModel.end());
      float dataModel = (float)*(dataMinusModel.begin()+dataMinusModel.size()/2);
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
      if (!_rmsBuffer.full()) {

      _rmsBuffer.push_back(spectrumRMS);
      _rmsRunAve = std::accumulate(_rmsBuffer.begin(), _rmsBuffer.end(), 0.0)/_rmsBuffer.size();
      
      }
      else {
	//just update the running average with the new value, and take the oldest off the end
	//	std::cout << "History buffer now full " << _num << std::endl;
	_rmsRunAve -= (_rmsBuffer.front()/_rmsBuffer.size());
	_rmsBuffer.push_back(spectrumRMS);
	_rmsRunAve += (_rmsBuffer.back()/_rmsBuffer.capacity());
      }
      

      // and update the model
      float k = 4;
      float meanMinusMinimum = k / sqrt(2.0*k); 
	
      dataModel = dataModel + meanMinusMinimum * _rmsRunAve;
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

	    //The following is for polarization
	    for(unsigned int pol = 0; pol < nPolarisations; ++pol ) {
	      long index = stokesAll->index(s, nSubbands,
					    pol, nPolarisations, t, nChannels );
	      I[index + c] = _lastGoodSpectrum[binLocal];
	      W[index +c] = 1.0;
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
      spectrumSum /= totalGoodChannels;

      // Check if more than 20% of of the channels in each band were
      // bad. If so in more than half of the bands, set a flag to
      // true.
      unsigned badBands = 0; 
      for (unsigned b = 0; b < 8; ++b){
	if (goodChannels[b] < 0.8 * channelsPerBand) {
	  if (++badBands == 4 ) break;
	}
      }

      //      std::cout << "Current model, mean and rms: " << _bandPass.mean() << " " << _bandPass.rms() << std::endl;
      //      std::cout << "Current data, mean and rms: " << spectrumSum << " " << spectrumRMS << std::endl;
      //      std::cout << "new dataModel: " << dataModel << std::endl;

      //      std::cout << "good channels " << goodChannels << std::endl;
      
      // Let us now build up the running average of spectrumSum values
      // (_maxHistory of them)
      // if the buffer is not full, compute the new meanRunAve like this


      if (!_meanBuffer.full()) {
	_meanBuffer.push_back(spectrumSum);
	_meanRunAve = std::accumulate(_meanBuffer.begin(), _meanBuffer.end(), 0.0)/_meanBuffer.size();
      }
      else {
	//   just update the running average with the new value, and take the oldest off the end
	//	std::cout << "History buffer now full " << _num << std::endl;
	_meanRunAve -= _meanBuffer.front()/_meanBuffer.size();
      	_meanBuffer.push_back(spectrumSum);
      	_meanRunAve += _meanBuffer.back()/_meanBuffer.size();
      }

      // Now we can check if this spectrum has an abnormally high mean
      // compared to the running average
      
      // Let us define the tolerance first, and remember, we are
      // summing across nBins, hence sqrt(nBins), strictly only valid
      // for Gaussian stats
      float spectrumRMStolerance = _srFactor * _bandPass.rms()/sqrt(nBins); 

      //Now check, if spectrumSum - model > tolerance, declare this
      //time sample useless, replace its data and take care of the
      //running averages, also cut the first 10 spectra, also cut
      //spectra where badBands == 4, see above

      if (spectrumSum - _meanRunAve > spectrumRMStolerance || _meanBuffer.size() < 10 || badBands == 4) {

	// we need to remove this entire spectrum
	clipSample( stokesAll, W, t, _lastGoodSpectrum );
	// now remove the last samples from the running averages
	//	_meanBuffer.pop_back();
	//	_rmsBuffer.pop_back();
      }
      // else keep a copy of the original spectrum, as it is good!
      else {
	for (unsigned s = 0; s < nSubbands; ++s) {
	  long index = stokesAll->index(s, nSubbands,
					0, nPolarisations,
					t, nChannels );
	  for (unsigned c = 0; c < nChannels; ++c) {
	    int binLocal = s*nChannels +c;
	    _lastGoodSpectrum[binLocal] = I[index+c];
	  }
	}
      }
      
      
      // Now we have a valid spectrum, either the original or
      // replaced; this spectrum is good, so let us do the final
      // bits of post processing reset the spectrumSum, and SumSq,
      // and flatten subtract the bandpass from the data.
      // 
      spectrumSum = 0.0; // SumSq is still zero
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

      // Avoid nastiness in those first 10 spectra by avoiding divisions by zero
      if (spectrumRMS == 0.0) spectrumRMS = 1.0;
      
      
      for (unsigned s = 0; s < nSubbands; ++s) {
	long index = stokesAll->index(s, nSubbands,
				      0, nPolarisations,
				      t, nChannels );
	for (unsigned c = 0; c < nChannels; ++c) {
	  I[index+c] -= _zeroDMing * spectrumSum;
	  I[index+c] /= spectrumRMS;
	}
      }
      // The bandpass is flat and the spectrum clean and normalized,
      // so move to the next spectrum
      // write out some stats:
      if (_num == 0) {
	std::cout << "RFI report ###############################" << std::endl;
	std::cout << "Running average of mean is at: " << _meanRunAve << std::endl;
	std::cout << "Running average of rms is at: " << _rmsRunAve << std::endl;
	std::cout << "###############################" << std::endl;
	std::cout << "#" << std::endl;
	std::cout << "#" << std::endl;
      }    
      // and update the model
      _bandPass.setMean(_meanRunAve);
      _bandPass.setRMS(_rmsRunAve);
      ++_num;
      _num = _num % _maxHistory;
      
    }
    // set the stats of the chunk
    weightedStokes->setRMS( _rmsRunAve );
    weightedStokes->setMean( _meanRunAve);
  }
}
} // namespace ampp
} // namespace pelican
