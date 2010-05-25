#include "ChanneliserPolyphase.h"

#include "pelican/utility/ConfigNode.h"
#include "TimeStreamData.h"
#include "ChannelisedStreamData.h"
#include <QtCore/QString>
#include <cstring>
#include <complex>
#include <iostream>
#include <fftw3.h>


using std::complex;

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

    // Create the fft plan.
    size_t fftSize = _nChannels * sizeof(fftw_complex);
    fftw_complex* in = (fftw_complex*) fftw_malloc(fftSize);
    fftw_complex* out = (fftw_complex*) fftw_malloc(fftSize);
    _fftPlan = fftw_plan_dft_1d(_nChannels, in, out, FFTW_FORWARD, FFTW_MEASURE);
    fftw_free(in);
    fftw_free(out);
}


/**
 * @details
 * Destroys the channeliser module.
 */
ChanneliserPolyphase::~ChanneliserPolyphase()
{
    fftw_destroy_plan(_fftPlan);
}



/**
 * @details
 * Method to run the channeliser.
 *
 * @param[in]  timeData Buffer of time samples to be channelised.
 * @param[out] spectrum Channels produced.
 */
void ChanneliserPolyphase::run(const TimeStreamData* timeData,
		const PolyphaseCoefficients* filterCoeff,
		ChannelisedStreamData* spectra)
{
	_checkData(timeData, filterCoeff);

	// TODO: Combine polarisations ...? (maybe do this in the adapter)
    unsigned nPolarisations = timeData->nPolarisations();
	const unsigned nSubbands = timeData->nSubbands();
    const unsigned nFilterTaps = filterCoeff->nTaps();

    unsigned bufferSize = _setupBuffers(nSubbands, _nChannels, nFilterTaps);

    // Resize the output spectra data blob.
	nPolarisations = 1;
    spectra->resize(nSubbands, nPolarisations, _nChannels);

    // Pointers to processing buffers.
    complex<double>* subbandBuffer;
    std::vector<complex<double> > filteredData(_nChannels);
    const complex<double>* coeff = filterCoeff->coefficients();

    for (unsigned s = 0; s < nSubbands; ++s) {

    	subbandBuffer = &(_subbandBuffer[s])[0];

    	// Update buffered (lagged) data for the subband.
        _updateBuffer(timeData->data(s), _nChannels, subbandBuffer, bufferSize);

        // Apply the PPF.
        _filter(subbandBuffer, nFilterTaps, _nChannels, coeff, &filteredData[0]);

        // FFT the filtered subband data to form a new spectrum.
        _fft(&filteredData[0], _nChannels, spectra->data(s));
    }
}


/**
 * @details
 */
void ChanneliserPolyphase::_checkData(const TimeStreamData* timeData,
		const PolyphaseCoefficients* filterCoeff)
{
	if (!timeData)
		throw QString("ChanneliserPolyphase: Time stream data blob missing.");

	if (!filterCoeff)
		throw QString("ChanneliserPolyphase: filter coefficients data blob missing.");

	if (timeData->nSubbands() == 0)
		throw QString("ChanneliserPolyphase: Empty time data blob");

	if (timeData->nSamples() == 0)
		throw QString("ChanneliserPolyphase: Empty time data blob");

	if (timeData->nSamples() == _nChannels)
		throw QString("ChanneliserPolyphase: Dimension mismatch: "
				"Number of samples %1 != number of output channels %2.")
				.arg(timeData->nSamples()).arg(_nChannels);

	if (filterCoeff->nChannels() != _nChannels)
		throw QString("ChanneliserPolyphase: Dimension mismatch: "
				"Number of filter channels %1 != number of output channels %2.")
				.arg(timeData->nSamples()).arg(_nChannels);
}


/**
 * @details
 * Sets up processing buffers
 *
 * @param nChannels
 * @param nFilterTaps
 */
unsigned ChanneliserPolyphase::_setupBuffers(const unsigned nSubbands,
		const unsigned nChannels, const unsigned nFilterTaps)
{
    _subbandBuffer.resize(nSubbands);
    unsigned bufferSize = nChannels * nFilterTaps;
    for (unsigned s = 0; s < nSubbands; ++s) {
        _subbandBuffer[s].resize(bufferSize, complex<double>(0.0, 0.0));
    }
    return bufferSize;
}

/**
 * @details
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
 * @details
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
    for (unsigned c = 0; c < nChannels; ++c) {
        for (unsigned t = 0; t < nTaps; ++t) {
//            unsigned iBuffer = (nTaps - t - 1) * nChannels + c;
            unsigned iCoeff = nTaps * c + t;
//            filteredSamples[c] += coefficients[iCoeff] * sampleBuffer[iBuffer];
            filteredSamples[c] += coefficients[iCoeff] * sampleBuffer[iCoeff];
        }
    }
}




/**
 * @details
 * FFT a vector of nSamples time data samples to produce a spectrum.
 *
 * @param samples
 * @param nSamples
 * @param spectrum
 */
void ChanneliserPolyphase::_fft(const complex<double>* samples,
        const unsigned nSamples, complex<double>* spectrum)
{
    fftw_execute_dft(_fftPlan, (fftw_complex*)samples, (fftw_complex*)spectrum);
}



}// namespace lofar
}// namespace pelican

