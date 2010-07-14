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
    QString window = config.getOption("filter", "filterWindow", "kaiser");
    QString coeffFile = config.getOption("filter", "fileName", "");

    // Load the coefficients.
    _ppfCoeffs.resize(nTaps, _nChannels);

    if (!coeffFile.isEmpty()) {
        if (!QFile::exists(coeffFile)) {
            throw QString("PPFChanneliser:: Unable to find coefficient file '%1'.")
            .arg(coeffFile);
        }
        _ppfCoeffs.load(coeffFile, nTaps, _nChannels);
    }
    else {
//        std::cout << "Generating coefficients..." << std::endl;
        window = window.toLower();
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
    }

//    const double* c = _ppfCoeffs.ptr();
//    unsigned nCoeffs = _ppfCoeffs.size();
//    _coeffs.resize(nCoeffs);
//    for (unsigned i = 0; i < nCoeffs; ++i) {
//        _coeffs[i] = float(c[i]);
//    }

    // As the channeliser currently only works for even number of channels
    // enforce this.
    if (_nChannels % 2 != 0) {
        throw QString("PPFChanneliser: "
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

    // Set up the buffers if required.
    unsigned nFilterTaps = _ppfCoeffs.nTaps();
    if (!_buffersInitialised) {
        _setupWorkBuffers(nSubbands, nPolarisations, _nChannels, nFilterTaps);
    }

    //const float* coeffs = &_coeffs[0];
    const double* coeffs = _ppfCoeffs.ptr();

    // Pointers to processing buffers.
    omp_set_num_threads(_nThreads);
    unsigned bufferSize = _workBuffer[0].size();

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
                        throw QString("PPFChanneliser::run(): dimension mismatch");
                    }

                    // Get a pointer to the work buffer.
                    workBuffer = &(_workBuffer[s * nPolarisations + p])[0];

                    // Update buffered (lagged) data for the sub-band.
                    _updateBuffer(timeData, _nChannels, workBuffer, bufferSize);

                    // Apply the PPF.
                    _filter(workBuffer, nFilterTaps, _nChannels, coeffs,
                            filteredSamples);

                    Spectrum<Complex>* spectrum = spectra->ptr(b, s, p);
                    spectrum->resize(_nChannels);
                    Complex* spectrumData = spectrum->ptr();

                    // FFT the filtered subband data to form a new spectrum.
                    _fft(filteredSamples, _nChannels, spectrumData);
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
        unsigned nSamples, Complex* buffer, unsigned bufferSize)
{
    Complex* dest = &buffer[nSamples];
    size_t size = (bufferSize - nSamples) * sizeof(Complex);
    memmove(dest, buffer, size);
    memcpy(buffer, samples, nSamples * sizeof(Complex));
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
        unsigned nChannels, const double* coeffs, Complex* filteredSamples)
{
    for (unsigned i = 0; i < nChannels; ++i) {
        filteredSamples[i] = Complex(0.0, 0.0);
    }

//#undef USE_CBLAS // undefine the use of cblas

    for (unsigned t = 0; t < nTaps; ++t) {
        for (unsigned c = 0; c < nChannels; ++c) {
            unsigned iBuffer = t * nChannels + c;
            unsigned iCoeff = t * nChannels + c;
            float re = sampleBuffer[iBuffer].real() * coeffs[iCoeff];
            float im = sampleBuffer[iBuffer].imag() * coeffs[iCoeff];
            filteredSamples[c] += std::complex<float>(re, im);

        }
    }

//    for (unsigned c = 0; c < nChannels; ++c) {
//        //#ifdef USE_CBLAS
//        //        unsigned iCoeff = c * nTaps;
//        //        unsigned iBuffer = (nTaps - 1) * nChannels + c;
//        //        //std::cout << c << " "<< iCoeff << " " << iBuffer << std::endl;
//        //        const Complex* x = &(sampleBuffer[iBuffer]);
//        //        // NOTE: coeffs are real and samples are complex!
//        //        const Complex* y = &(coeffs[iCoeff]);
//        //        Complex result;
//        //        cblas_cdotu_sub(nTaps, x, -64, y, 1, &result);
//        //        filteredSamples[c] = result;
//        //#else
//        for (unsigned t = 0; t < nTaps; ++t) {
//            //unsigned iBuffer = (nTaps - t - 1) * nChannels + c;
//            unsigned iBuffer = t * nChannels + c;
//            unsigned iCoeff = nTaps * c + t;
//            float re = sampleBuffer[iBuffer].real() * coeffs[iCoeff];
//            float im = sampleBuffer[iBuffer].imag() * coeffs[iCoeff];
//            filteredSamples[c] += std::complex<float>(re, im);
//        }
//        //#endif
//    }
}


/**
* @details
* FFT a vector of nSamples time data samples to produce a spectrum.
*
* @param samples
* @param nSamples
* @param spectrum
*/
void PPFChanneliser::_fft(const Complex* samples, unsigned nSamples,
        Complex* spectrum)
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
void PPFChanneliser::_fftShift(Complex* spectrum, unsigned nChannels)
{
    std::vector<Complex> temp(nChannels, Complex(0.0, 0.0));
    unsigned iZero = nChannels / 2; // FIXME? only works for even nChannels!
    size_t size = nChannels / 2 * sizeof(Complex);
    memcpy(&temp[iZero], spectrum,  size);
    memcpy(&temp[0], &spectrum[iZero],  size);
    memcpy(spectrum, &temp[0], nChannels * sizeof(Complex));
}

/**
 * @details
 * Calculates start and end indices for a given thread for splitting nValues
 * of a processing dimension between nThreads.
 *
 * @param start
 * @param end
 * @param nValues
 * @param nThreads
 * @param threadId
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
* Sets up processing buffers
*
* @param nChannels
* @param nFilterTaps
*/
unsigned PPFChanneliser::_setupWorkBuffers(unsigned nSubbands,
        unsigned nPolarisations, unsigned nChannels, unsigned nTaps)
{
    _workBuffer.resize(nSubbands * nPolarisations);
    unsigned bufferSize = nChannels * nTaps;
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

