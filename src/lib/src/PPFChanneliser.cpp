#include "PPFChanneliser.h"

#include "pelican/utility/ConfigNode.h"

#include "TimeSeries.h"
#include "SubbandSpectra.h"

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
PPFChanneliser::PPFChanneliser(const ConfigNode& config)
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
    size_t fftSize = _nChannels * sizeof(fftwf_complex);
    fftwf_complex* in = (fftwf_complex*) fftwf_malloc(fftSize);
    fftwf_complex* out = (fftwf_complex*) fftwf_malloc(fftSize);
    _fftPlan = fftwf_plan_dft_1d(_nChannels, in, out, FFTW_FORWARD, FFTW_MEASURE);
    fftwf_free(in);
    fftwf_free(out);
}


/**
* @details
* Destroys the channeliser module.
*/
PPFChanneliser::~PPFChanneliser()
{
    fftwf_destroy_plan(_fftPlan);
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
void PPFChanneliser::run(const TimeSeriesC32* timeSeries,
        SubbandSpectraC32* spectra)
{
    _checkData(timeSeries);

//  TODO: Combine polarisations ...? (maybe do this in the adapter)
//	unsigned nPolarisations = timeData->nPolarisations();
//	unsigned bufferSize = _setupBuffers(nSubbands, _nChannels, nFilterTaps);

    // Get local copies of the data dimensions.
    unsigned nSubbands = timeSeries->nSubbands();
    unsigned nPolarisations = timeSeries->nPolarisations();
    unsigned nTimes = timeSeries->nSamples();
    unsigned nTimeBlocks = _nChannels / nTimes; // TODO: this must be an int ratio.

    // Resize the output spectra blob.
    spectra->resize(nTimeBlocks, nSubbands, nPolarisations);

    // Set up the buffers if required.
    unsigned nFilterTaps = _coeffs.nTaps();
    if (!_buffersInitialised) {
        _setupBuffers(nSubbands, _nChannels, nFilterTaps);
    }

    // Pointers to processing buffers.
    omp_set_num_threads(_nThreads);
    unsigned bufferSize = _subbandBuffer[0].size();
    const double* coeff = _coeffs.coefficients();

#pragma omp parallel
    {
        unsigned threadId = omp_get_thread_num();
        unsigned start = 0, end = 0;
        _threadSubbandRange(start, end, nSubbands, _nThreads, threadId);

        complex<float>* subbandBuffer;
        complex<float>* filteredSamples = &(_filteredData[threadId])[0];

        // TODO: loop over time blocks and polarisations.

        // Loop over sub-bands.
        for (unsigned b = 0; b < nTimeBlocks; ++b) {
            for (unsigned s = start; s < end; ++s) {
                for (unsigned p = 0; p < nPolarisations; ++p) {

                    // Get a pointer to the work buffer for the sub-band being
                    // processed.
                    subbandBuffer = &(_subbandBuffer[s])[0];

                    const complex<float>* times = timeSeries->ptr(s, p);

                    // Update buffered (lagged) data for the sub-band.
                    _updateBuffer(times, _nChannels,
                            subbandBuffer, bufferSize);

                    // Apply the PPF.
                    _filter(subbandBuffer, nFilterTaps, _nChannels, coeff,
                            filteredSamples);

                    Spectrum<std::complex<float> >* spectrum =
                            spectra->ptr(b, s, p);

                    spectrum->resize(_nChannels);

                    std::complex<float>* channelAmps = spectrum->ptr();

                    // FFT the filtered subband data to form a new spectrum.
                    _fft(filteredSamples, _nChannels, channelAmps);
                }
            }
        }

    } // end of parallel region
}


/**
* @details
*/
void PPFChanneliser::_checkData(const TimeSeriesC32* timeData)
{
    if (!timeData)
        throw QString("ChanneliserPolyphase: Time stream data blob missing.");

    if (timeData->nSubbands() == 0)
        throw QString("ChanneliserPolyphase: Empty time data blob");

    if (timeData->nSamples() == 0)
        throw QString("ChanneliserPolyphase: Empty time data blob");

    if (timeData->nSamples() != _nChannels)
        throw QString("ChanneliserPolyphase: Dimension mismatch: "
                "Number of samples %1 != number of output channels %2.")
                .arg(timeData->nSamples()).arg(_nChannels);

    if (_coeffs.nChannels() != _nChannels)
        throw QString("ChanneliserPolyphase: Dimension mismatch: "
                "Number of filter channels %1 != number of output channels %2.")
                .arg(_coeffs.nChannels()).arg(_nChannels);
}


/**
* @details
* Prepend nSamples complex data into the start of the buffer moving along
* other data.
*
* @param samples
* @param nSamples
*/
void PPFChanneliser::_updateBuffer(const complex<float>* samples,
        unsigned nSamples, complex<float>* buffer, unsigned bufferSize)
{
    complex<float>* dest = &buffer[nSamples];
    size_t size = (bufferSize - nSamples) * sizeof(complex<float>);
    memmove(dest, buffer, size);
    memcpy(buffer, samples, nSamples * sizeof(complex<float>));
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
void PPFChanneliser::_filter(const complex<float>* sampleBuffer,
        unsigned nTaps, unsigned nChannels,
        const double* coeffs, complex<float>* filteredSamples)
{
    for (unsigned i = 0; i < nChannels; ++i) {
        filteredSamples[i] = complex<float>(0.0, 0.0);
    }

    for (unsigned c = 0; c < nChannels; ++c) {
        for (unsigned t = 0; t < nTaps; ++t) {
            unsigned iBuffer = (nTaps - t - 1) * nChannels + c;
            unsigned iCoeff = nTaps * c + t;
            float C = coeffs[iCoeff];
            complex<float> value = sampleBuffer[iBuffer];
            filteredSamples[c] += C * value;
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
void PPFChanneliser::_fft(const complex<float>* samples,
        unsigned nSamples, complex<float>* spectrum)
{
    fftwf_execute_dft(_fftPlan, (fftwf_complex*)samples, (fftwf_complex*)spectrum);
    _fftShift(spectrum, nSamples);
}


/**
 * @details
 * Shift the zero frequency component to the centre of the spectrum.
 *
 * @param spectrum
 * @param nChannels
 */
void PPFChanneliser::_fftShift(complex<float>* spectrum,
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
void PPFChanneliser::_threadSubbandRange(unsigned& start,
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
unsigned PPFChanneliser::_setupBuffers(unsigned nSubbands,
        unsigned nChannels, unsigned nFilterTaps)
{
    _subbandBuffer.resize(nSubbands);
    unsigned bufferSize = nChannels * nFilterTaps;
    for (unsigned s = 0; s < nSubbands; ++s) {
        _subbandBuffer[s].resize(bufferSize, complex<float>(0.0, 0.0));
    }
    _buffersInitialised = true;
    return bufferSize;
}


}// namespace lofar
}// namespace pelican

