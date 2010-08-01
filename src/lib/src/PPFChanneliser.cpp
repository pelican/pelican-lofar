#include "PPFChanneliser.h"

#include "pelican/utility/ConfigNode.h"

#include "SubbandTimeSeries.h"
#include "TimeSeries.h"
#include "SubbandSpectra.h"

#include <QtCore/QString>
#include <QtCore/QTime>
#include <QtCore/QFile>

#include <cstring>
#include <complex>
#include <iostream>

#include <fftw3.h>
#include <omp.h>

//#ifdef USING_MKL
//    #include <mkl.h>
//    #define USE_CBLAS
//#else
//    extern "C" {
//        #include <cblas.h>
//    }
//    #define USE_CBLAS
//#endif


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

    // Get options from the XML configuration node.
    _nChannels = config.getOption("channels", "number", "512").toUInt();
    _nThreads = config.getOption("processingThreads", "number", "2").toUInt();
    unsigned nTaps = config.getOption("coefficients", "nTaps", "8").toUInt();
    QString window = config.getOption("filter", "filterWindow", "kaiser").toLower();

    // Enforce even number of channels.
    if (_nChannels % 2 != 0) {
        throw QString("PPFChanneliser: "
                " Please choose an even number of channels.");
    }

    // Generate PPF coefficients.
    _ppfCoeffs.resize(nTaps, _nChannels);
    PolyphaseCoefficients::FirWindow windowType;
    if (window == "kaiser") {
        windowType = PolyphaseCoefficients::KAISER;
    }
    else if (window == "gaussian") {
        windowType = PolyphaseCoefficients::GAUSSIAN;
    }
    else if (window == "blackman") {
        windowType = PolyphaseCoefficients::BLACKMAN;
    }
    else if (window == "hamming") {
        windowType = PolyphaseCoefficients::HAMMING;
    }
    else {
        throw QString("PPFChanneliser: "
                "Unknown coefficient window type '%1'.").arg(window);
    }
    _ppfCoeffs.genereateFilter(nTaps, _nChannels, windowType);

    // Convert Coefficients to single precision.
    _coeffs.resize(_ppfCoeffs.size());
    const double* coeffs = _ppfCoeffs.ptr();
    for (unsigned i = 0; i < _ppfCoeffs.size(); ++i) {
        _coeffs[i] = float(coeffs[i]);
    }

    // Allocate buffers used for holding the output of the FIR stage.
    _filteredData.resize(_nThreads);
    for (unsigned i = 0; i < _nThreads; ++i) {
        _filteredData[i].resize(_nChannels);
    }

    // Initialise pointer to the current oldest sample set.
    _iOldestSamples = 0;

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
void PPFChanneliser::run(const SubbandTimeSeriesC32* timeSeries,
        SubbandSpectraC32* spectra)
{
    _checkData(timeSeries);

    // Get local copies of the data dimensions.
    unsigned nSubbands = timeSeries->nSubbands();
    unsigned nPolarisations = timeSeries->nPolarisations();
    unsigned nTimeBlocks = timeSeries->nTimeBlocks();

    // Resize the output spectra blob.
    spectra->resize(nTimeBlocks, nSubbands, nPolarisations);

    // Set the timing parameters
    // We only need the timestamp of the first packet for this version of the Channeliser
    spectra -> setLofarTimestamp(timeSeries -> getLofarTimestamp());
    spectra -> setBlockRate(timeSeries -> getBlockRate());

    // Set up the buffers if required.
    unsigned nFilterTaps = _ppfCoeffs.nTaps();
    if (!_buffersInitialised) {
        _setupWorkBuffers(nSubbands, nPolarisations, _nChannels, nFilterTaps);
    }

//    std::cout << std::endl;
//    std::cout << "nSubbands = " << nSubbands << std::endl;
//    std::cout << "nPolarisations = " << nPolarisations << std::endl;
//    std::cout << "nTimeBlocks = " << nTimeBlocks << std::endl;
//    std::cout << "nFilterTaps = " << nFilterTaps << std::endl;

    const float* coeffs = &_coeffs[0];

    // Pointers to processing buffers.
    omp_set_num_threads(_nThreads);

#pragma omp parallel
    {
        unsigned threadId = omp_get_thread_num();
        unsigned start = 0, end = 0;
        _threadProcessingIndices(start, end, nTimeBlocks, _nThreads, threadId);

        Complex* workBuffer;
        Complex* filteredSamples = &(_filteredData[threadId])[0];

        // Loop over sub-bands.
        for (unsigned b = start; b < end; ++b) {
            for (unsigned s = 0; s < nSubbands; ++s) {
                for (unsigned p = 0; p < nPolarisations; ++p) {

                    // Get a pointer to the time series.
                    const TimeSeries<Complex>* times = timeSeries->ptr(b,s,p);
                    const Complex* timeData = times->ptr();
                    unsigned nTimes = times->nTimes();

                    if (nTimes != _nChannels) {
                        std::cout << "nTimes: " << nTimes << " nChannels: " << _nChannels << std::endl;
                        throw QString("PPFChanneliser::run(): dimension mismatch");
                    }

                    // Get a pointer to the work buffer.
                    workBuffer = &(_workBuffer[s * nPolarisations + p])[0];

//                    Update buffered (lagged) data for the sub-band.
                    _updateBuffer(timeData, _nChannels, nFilterTaps,  workBuffer);

                    // Apply the PPF.
                    _filter(workBuffer, nFilterTaps, _nChannels, coeffs, filteredSamples);

                    Spectrum<Complex>* spectrum = spectra->ptr(b, s, p);
                    spectrum->resize(_nChannels);
                    Complex* spectrumData = spectrum->ptr();

                    // FFT the filtered subband data to form a new spectrum.
                    _fft(filteredSamples, spectrumData);
                }
            }
        }

    } // end of parallel region
}


/**
* @details
*/
void PPFChanneliser::_checkData(const SubbandTimeSeriesC32* timeData)
{
    if (!timeData)
        throw QString("PPFChanneliser: Time stream data blob missing.");

    if (timeData->size() == 0)
        throw QString("PPFChanneliser: Empty time data blob");

    if (timeData->nSubbands() == 0)
        throw QString("PPFChanneliser: Empty time data blob");

    if (timeData->nPolarisations() == 0)
        throw QString("PPFChanneliser: Empty time data blob");

    if (timeData->nTimeBlocks() == 0)
        throw QString("PPFChanneliser: Empty time data blob");

    if (_ppfCoeffs.nChannels() != _nChannels)
        throw QString("PPFChanneliser: Dimension mismatch: "
                "Number of filter channels %1 != number of output channels %2.")
                .arg(_ppfCoeffs.nChannels()).arg(_nChannels);
}


/**
* @details
* Prepend nSamples complex data into the start of the buffer moving along
* other data.
*
* @param samples
* @param nSamples
*/
void PPFChanneliser::_updateBuffer(const Complex* samples,
        unsigned nSamples, unsigned nTaps, Complex* buffer)
{
    size_t blockSize = nSamples * sizeof(Complex);
    memcpy(&buffer[_iOldestSamples * nSamples], samples, blockSize);
    _iOldestSamples = (_iOldestSamples + 1) % nTaps;
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
void PPFChanneliser::_filter(const Complex* sampleBuffer, unsigned nTaps,
        unsigned nChannels, const float* coeffs, Complex* filteredSamples)
{
    for (unsigned i = 0; i < nChannels; ++i) {
        filteredSamples[i] = Complex(0.0, 0.0);
    }

    unsigned iBuffer = 0, idx = 0;
    float re, im, coeff;

    for (unsigned i = 0, t = 0; t < nTaps; ++t) {
        iBuffer = ((_iOldestSamples + t) % nTaps) * nChannels;
        for (unsigned c = 0; c < nChannels; ++c) {
            idx = iBuffer + c;
            coeff = coeffs[i];
            re = sampleBuffer[idx].real() * coeff;
            im = sampleBuffer[idx].imag() * coeff;
            filteredSamples[c].real() += re;
            filteredSamples[c].imag() += im;
            i++;
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
void PPFChanneliser::_fft(const Complex* samples, Complex* spectrum)
{
    fftwf_execute_dft(_fftPlan, (fftwf_complex*)samples, (fftwf_complex*)spectrum);
}


/**
 * @details
 * Calculates start and end indices for a given thread for splitting nValues
 * of a processing dimension between nThreads.
 */
void PPFChanneliser::_threadProcessingIndices(unsigned& start,
        unsigned& end, unsigned nValues, unsigned nThreads,
        unsigned threadId)
{
    if (threadId >= nThreads) {
        throw QString("PPFChanneliser::_threadProcessingIndices(): "
                "threadId '%1' out of range for nThreads = %2.").arg(threadId).
                arg(nThreads);
    }

    if (threadId >= nValues) {
        start = end = 0;
        return;
    }

    unsigned number = nValues / nThreads;
    unsigned remainder = nValues % nThreads;
    if (threadId < remainder)
        number++;

    start = threadId * number;
    if (threadId >= remainder) start += remainder;
    end = start + number;
}


/**
* @details
* Set up buffers used to store the last nTaps * nChannels time series values
* for each sub-band and polarisation.
*/
unsigned PPFChanneliser::_setupWorkBuffers(unsigned nSubbands,
        unsigned nPolarisations, unsigned nChannels, unsigned nTaps)
{
    unsigned bufferSize = nChannels * nTaps;
    _workBuffer.resize(nSubbands * nPolarisations);
    for (unsigned i = 0, s = 0; s < nSubbands; ++s) {
        for (unsigned p = 0; p < nPolarisations; ++p) {
            _workBuffer[i].resize(bufferSize, Complex(0.0, 0.0));
            i++;
        }
    }
    _buffersInitialised = true;
    return bufferSize;
}


}// namespace lofar
}// namespace pelican

