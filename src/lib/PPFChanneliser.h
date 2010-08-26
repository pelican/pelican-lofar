#ifndef PPF_CHANNELISER_H_
#define PPF_CHANNELISER_H_

/**
 * @file PPFChanneliser.h
 */

#include "pelican/modules/AbstractModule.h"
#include "PolyphaseCoefficients.h"

#include <complex>
#include <vector>

#include <fftw3.h>

using std::vector;
using std::cout;
using std::cerr;
using std::endl;

namespace pelican {

class ConfigNode;

namespace lofar {

class TimeSeriesDataSetC32;
class SpectrumDataSetC32;

/**
 * @class PPFChanneliser
 *
 * @brief
 * Module to channelise a time stream data blob.
 *
 * @details
 * Channelises time stream data using a polyphase channelising filter.
 *
 * \section Configuration:
 *
 * Example configuration node.
 *
 * \verbatim
 * 		<PPFChanneliser name="">
 * 			<channels number="512"/>
 * 			<processingThreads number="2"/>
 * 			<filter nTaps="8" filterWindow="kaiser"/>
 * 		</PPFChanneliser>
 * \verbatim
 *
 * - channels: The number of channels generated in the spectra.
 * - processingThreads: The number of threads to parallelise over.
 * - filter: Options for FIR filer coefficients.
 *     - nTaps: Number of filter taps in the PPF coefficient data
 *     - filterWindow: The filter window type used in generating FIR filer
 *       coefficients. Possible options are: "kaiser" (default), "gaussian",
 *       "blackman" and "hamming".
 *
 */

class PPFChanneliser : public AbstractModule
{
    private:
        friend class PPFChanneliserTest;
        typedef std::complex<float> Complex;

    public:
        /// Constructs the channeliser module.
        PPFChanneliser(const ConfigNode& config);

        /// Destroys the channeliser module.
        ~PPFChanneliser();

        /// Method converting the time stream to a spectrum.
        void run(const TimeSeriesDataSetC32* timeSeries,
                SpectrumDataSetC32* spectra);

    private:
        /// Generate the FIR coefficients used by the PPF.
        void _generateFIRCoefficients(const QString& window, unsigned nTaps);

        /// Sanity checking.
        void _checkData(const TimeSeriesDataSetC32* timeData);

        /// Update the sample buffer.
        void _updateBuffer(const Complex* samples, unsigned nSamples,
                unsigned nTaps, Complex* buffer);

        /// Filter the matrix of samples (dimensions nTaps by nChannels)
        /// to create a vector of samples for the FFT.
        void _filter(const Complex* sampleBuffer, unsigned nTaps,
                unsigned nChannels, const float* coeffs,
                Complex* filteredSamples);

        /// FFT filtered samples to form a spectrum.
        void _fft(const Complex* samples, Complex* spectrum);

        /// Returns the sub-band ID range to be processed.
        void _threadProcessingIndices(unsigned& start, unsigned& end,
                unsigned nSubbands, unsigned nThreads, unsigned threadId);

        /// Set up processing buffers.
        unsigned _setupWorkBuffers(unsigned nSubbands,
                unsigned nPolariations, unsigned nChannels, unsigned nTaps);

        /// Return an error message.
        QString _err(const QString& message);

    private:
        bool _buffersInitialised;

        unsigned _nChannels;
        unsigned _nThreads;

        PolyphaseCoefficients _ppfCoeffs;
        vector<float> _coeffs;

        unsigned _iOldestSamples; // Pointer to the oldest samples.

        fftwf_plan _fftPlan;

        // Work Buffers (need to have a buffer per thread).
        vector<vector<Complex> > _workBuffer;
        vector<vector<Complex> > _filteredData;
};


/**
 * @details
 * FFT a vector of nSamples time data samples to produce a spectrum.
 *
 * @param samples
 * @param nSamples
 * @param spectrum
 */
inline void PPFChanneliser::_fft(const Complex* samples, Complex* spectrum)
{
    fftwf_execute_dft(_fftPlan, (fftwf_complex*)samples, (fftwf_complex*)spectrum);
}


// Declare this class as a pelican module.
PELICAN_DECLARE_MODULE(PPFChanneliser)

}// namespace lofar
}// namespace pelican

#endif // PPF_CHANNELISER_H_
