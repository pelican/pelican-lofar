#include "ChanneliserPolyphase.h"

#include "pelican/utility/ConfigNode.h"

#include "TimeStreamData.h"
#include "ChannelisedStreamData.h"

#include <QtCore/QString>
#include <QtCore/QTime>

#include <cstring>
#include <complex>
#include <iostream>

#include <fftw3.h>
#include <omp.h>

using std::complex;

namespace pelican {
namespace lofar {


/**
 * @details
 * Constructor.
 *
 * @param[in] config XML configuration node.
 */
ChanneliserPolyphase::ChanneliserPolyphase(const ConfigNode& config)
: AbstractModule(config)
{
    _buffersInitialised = false;

    // Get options from the config.
    _nChannels = config.getOption("channels", "number", "512").toUInt();
    _nThreads = config.getOption("processingThreads", "number", "2").toUInt();

    if (_nChannels % 2 != 0) {
        throw QString("ChanneliserPolyphase: "
                " Please choose an even number of channels.");
    }

    // Allocate buffers used for holding the output of the FIR stage.
    _filteredData.resize(_nThreads);
    for (unsigned i = 0; i < _nThreads; ++i) {
        _filteredData[i].resize(_nChannels);
    }

    // Create the FFTW plan.
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
* The channeliser performs channelisation of a number of sub-bands containing
* a complex time series.
*
* Paralellisation, by means of openMP threads, is carried out by splitting
* the subbands as evenly as possible between threads.
*
*
*
* @param[in]  timeData 		Buffer of time samples to be channelised.
* @param[in]  filterCoeff	Pointer to object containing polyphase filter
* 						 	coefficients.
* @param[out] spectrum	 	Set of spectra produced.
*/
void ChanneliserPolyphase::run(const TimeStreamData* timeData,
        const PolyphaseCoefficients* filterCoeff,
        ChannelisedStreamData* spectra)
{
    _checkData(timeData, filterCoeff);

//  TODO: Combine polarisations ...? (maybe do this in the adapter)
//	unsigned nPolarisations = timeData->nPolarisations();
//	unsigned bufferSize = _setupBuffers(nSubbands, _nChannels, nFilterTaps);

    // Resize the output spectra data blob.
    const unsigned nSubbands = timeData->nSubbands();
    const unsigned nPolarisations = 1;
    spectra->resize(nSubbands, nPolarisations, _nChannels);

    // Set up the buffers if required
    const unsigned nFilterTaps = filterCoeff->nTaps();
    if (!_buffersInitialised)
        _setupBuffers(nSubbands, _nChannels, nFilterTaps);

    // Set the timing parameters
    // We only need the timestamp of the first packet for this version of the Channeliser
    spectra -> setLofarTimestamp(timeData -> getLofarTimestamp());
    spectra -> setBlockRate(timeData -> getBlockRate());
    
    // Pointers to processing buffers.
    omp_set_num_threads(_nThreads);
    const unsigned bufferSize = _subbandBuffer[0].size();
    const double* coeff = filterCoeff->coefficients();

#pragma omp parallel
    {
        unsigned threadId = omp_get_thread_num();
        unsigned start = 0, end = 0;
        _threadSubbandRange(start, end, nSubbands, _nThreads, threadId);

        complex<double>* subbandBuffer;
        complex<double>* filteredSamples = &(_filteredData[threadId])[0];

        for (unsigned s = start; s < end; ++s) {
    
            // Get a pointer to the work buffer for the sub-band being processed.
            subbandBuffer = &(_subbandBuffer[s])[0];

            // Update buffered (lagged) data for the sub-band.
            _updateBuffer(timeData->data(s), _nChannels, subbandBuffer, bufferSize);

            // Apply the PPF.
            _filter(subbandBuffer, nFilterTaps, _nChannels, coeff, filteredSamples);

            // FFT the filtered subband data to form a new spectrum.
            _fft(filteredSamples, _nChannels, spectra->data(s));
        }

    } // end of parallel region
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

    if (timeData->nSamples() != _nChannels)
        throw QString("ChanneliserPolyphase: Dimension mismatch: "
                "Number of samples %1 != number of output channels %2.")
                .arg(timeData->nSamples()).arg(_nChannels);

    if (filterCoeff->nChannels() != _nChannels)
        throw QString("ChanneliserPolyphase: Dimension mismatch: "
                "Number of filter channels %1 != number of output channels %2.")
                .arg(filterCoeff->nChannels()).arg(_nChannels);
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
        unsigned nSamples, complex<double>* buffer, unsigned bufferSize)
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
        unsigned nTaps, unsigned nChannels,
        const double* coefficients, complex<double>* filteredSamples)
{
    for (unsigned i = 0; i < nChannels; ++i) {
        filteredSamples[i] = std::complex<double>(0.0, 0.0);
    }

    for (unsigned c = 0; c < nChannels; ++c) {
        for (unsigned t = 0; t < nTaps; ++t) {
            unsigned iBuffer = (nTaps - t - 1) * nChannels + c;
            unsigned iCoeff = nTaps * c + t;
            filteredSamples[c] += coefficients[iCoeff] * sampleBuffer[iBuffer];
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
        unsigned nSamples, complex<double>* spectrum)
{
//	for (unsigned i = 0; i < nSamples; ++i) {
//		spectrum[i] = std::complex<double>(0.0, 0.0);
//	}

    fftw_execute_dft(_fftPlan, (fftw_complex*)samples, (fftw_complex*)spectrum);
    _fftShift(spectrum, nSamples);

    // Normalise
//	for (unsigned c = 0; c < nSamples; ++c) {
//		spectrum[c] /= nSamples;
//	}
}


/**
 * @details
 * Shift the zero frequency component to the centre of the spectrum.
 *
 * @param spectrum
 * @param nChannels
 */
void ChanneliserPolyphase::_fftShift(complex<double>* spectrum,
        unsigned nChannels)
{
    std::vector<std::complex<double> > temp(nChannels,
            std::complex<double>(0.0, 0.0));
    unsigned iZero = nChannels / 2; // FIXME? only works for even nChannels!
    size_t size = nChannels / 2 * sizeof(complex<double>);
    memcpy(&temp[iZero], spectrum,  size);
    memcpy(&temp[0], &spectrum[iZero],  size);
    memcpy(spectrum, &temp[0], nChannels * sizeof(complex<double>));
}


/**
 * @details
 * Returns the subband range to process for a given number of subbands,
 * number of threads and thread Id.
 *
 * @param start
 * @param end
 * @param threadId
 * @param nTreads
 */
void ChanneliserPolyphase::_threadSubbandRange(unsigned& start,
        unsigned& end, unsigned nSubbands, unsigned nThreads,
        unsigned threadId)
{
    if (threadId >= nThreads) {
        throw QString("ChanneliserPolyphase::_threadSubbandRange(): "
                "threadId '%1' out of range for nThreads = %2.").arg(threadId).
                arg(nThreads);
    }

    if (threadId >= nSubbands) {
        start = end = 0;
        return;
    }

    unsigned number = nSubbands / nThreads;
    unsigned remainder = nSubbands % nThreads;
    if (threadId < remainder)
        number++;

    start = threadId * number;
    if (threadId >= remainder) start += remainder;
    end = start + number;
}


/**
* @details
* Sets up processing buffers
*
* @param nChannels
* @param nFilterTaps
*/
unsigned ChanneliserPolyphase::_setupBuffers(unsigned nSubbands,
        unsigned nChannels, unsigned nFilterTaps)
{
    _subbandBuffer.resize(nSubbands);
    unsigned bufferSize = nChannels * nFilterTaps;
    for (unsigned s = 0; s < nSubbands; ++s) {
        _subbandBuffer[s].resize(bufferSize, complex<double>(0.0, 0.0));
    }
    _buffersInitialised = true;
    return bufferSize;
}


}// namespace lofar
}// namespace pelican

