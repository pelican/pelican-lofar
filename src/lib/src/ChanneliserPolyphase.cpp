#include "ChanneliserPolyphase.h"

#include "TimeStreamData.h"
#include "ChannelisedStreamData.h"
#include <QtCore/QString>
#include <cstring>

#include "pelican/utility/ConfigNode.h"



namespace pelican {
namespace lofar {

// Declare this class as a pelican module.
PELICAN_DECLARE_MODULE(ChanneliserPolyphase)


/**
 * @details
 * Constructor.
 *
 * @param[in] config XML configuration node.
 */
ChanneliserPolyphase::ChanneliserPolyphase(const ConfigNode& config)
: AbstractModule(config)
{
    // Get options from the config.
	_nChannels = config.getOption("channels", "number", "512").toUInt();
	_nFilterTaps = config.getOption("filter", "nTaps", "8").toUInt();
	_nSubbands = config.getOption("subbands", "number", "62").toUInt();

	unsigned bufferSize = _nChannels * _nFilterTaps;
	for (unsigned s = 0; s < _nSubbands; ++s) {
		_subbandBuffer[s].resize(bufferSize, complex<double>(0.0, 0.0));
		_filteredBuffer[s].resize(_nChannels, complex<double>(0.0, 0.0));
	}

	// Create the fft plan
	_fftPlan = fftw_plan_dft_1d(_nChannels, _fftwIn, _fftwOut,
			FFTW_FORWARD, FFTW_MEASURE);

	_spectrum->resize(_nSubbands, 1, _nChannels);

	/// TODO
//	_filterCoeff.load("coeffs.dat");
	_filterCoeff.resize(_nFilterTaps, _nChannels);
}


/**
 * @details
 * Method to run the channeliser.
 *
 * @param[in]  timeData Buffer of time samples to be channelised.
 * @param[out] spectrum Channels produced.
 */
void ChanneliserPolyphase::run(const TimeStreamData* timeData,
		ChannelisedStreamData* spectrum)
{
	if (!timeData)
		throw QString("ChanneliserPolyphase: Time stream data blob missing.");

	if (!spectrum)
		throw QString("ChanneliserPolyphase: Spectrum data blob missing.");

	if (timeData->nSubbands() == 0)
		throw QString("ChanneliserPolyphase: Empty time data blob");

	if (timeData->nSamples() == 0)
			throw QString("ChanneliserPolyphase: Empty time data blob");

	if (timeData->nSamples() == _nChannels)
		throw QString("ChanneliserPolyphase: Dimension mismatch: "
				"Number of samples %1 != number of channels %2.")
				.arg(timeData->nSamples()).arg(_nChannels);

	// Add polarisations together? (maybe do this in the adapter)
	unsigned nSubbands = timeData->nSubbands();
	unsigned nPoarisations = timeData->nPolarisations();
	unsigned bufferSize = _subbandBuffer.size();

	// TODO: Channelise one subband at a time.
	for (unsigned s = 0; s < _nSubbands; ++s) {
		complex<double>* sampleBuffer = &(_subbandBuffer[s])[0];
		complex<double>* filteredBuffer = &(_filteredBuffer[s])[0];
		const complex<double>* coeff = _filterCoeff.coefficients();
		const complex<double>* newSamples = timeData->data(s);

		_updateBuffer(newSamples, _nChannels, sampleBuffer, bufferSize);
		_filter(sampleBuffer, _nFilterTaps, _nChannels, coeff, filteredBuffer);

		// get the vector to FFT here.
		complex<double>* spectrum = _spectrum->data(s);
		_fft(filteredBuffer, _nChannels, spectrum);
	}
}

/**
 * Prepend nSamples complex data into the start of the buffer moving along
 * other data.
 *
 * @param samples
 * @param nSamples
 */
void ChanneliserPolyphase::_updateBuffer(const complex<double>* samples,
		const unsigned nSamples, complex<double>* buffer, const unsigned bufferSize)
{
	complex<double>* dest = &buffer[nSamples];
	size_t size = (bufferSize - nSamples) * sizeof(complex<double>);
    memmove(dest, buffer, size);
    memcpy(buffer, samples, nSamples * sizeof(complex<double>));
}


/**
 * Filter a buffer of time samples.
 *
 * @param samples
 * @param nTaps
 * @param nChannels
 * @param filteredSamples
 */
void ChanneliserPolyphase::_filter(const complex<double>* sampleBuffer,
		const unsigned nTaps, const unsigned nChannels,
		const complex<double>* coefficients, complex<double>* filteredSamples)
{
	for (unsigned c = 0; nChannels; ++c) {
		for (unsigned t = 0; t < nTaps; ++t) {
			unsigned iBuffer = (nTaps - t) * nChannels + c;
			unsigned iCoeff = nTaps * c + t;
			filteredSamples[c] += coefficients[iCoeff] * sampleBuffer[iBuffer];
		}
	}
}




/**
 * FFT a vector of nSamples time data samples to produce a spectrum.
 *
 * @param samples
 * @param nSamples
 * @param spectrum
 */
void ChanneliserPolyphase::_fft(const complex<double>* samples,
		const unsigned nSamples, complex<double>* spectrum)
{
    _fftwIn = (fftw_complex*)samples;
    _fftwOut = (fftw_complex*)spectrum;
    fftw_execute(_fftPlan);
}



}// namespace lofar
}// namespace pelican

