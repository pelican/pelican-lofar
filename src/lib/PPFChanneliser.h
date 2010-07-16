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

using std::complex;

namespace pelican {

class ConfigNode;

namespace lofar {

class SubbandTimeSeriesC32;
class SubbandSpectraC32;

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
 * 		<ChanneliserPolyphase name="">
 * 			<channels number="512"/>
 * 			<processingThreads number="2"/>
 * 			<filter nTaps="8" filterWindow="kaiser"/>
 * 		</ChanneliserPolyphase>
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
        void run(const SubbandTimeSeriesC32* timeSeries,
                SubbandSpectraC32* spectra);

    private:
        /// Sainity checking.
        void _checkData(const SubbandTimeSeriesC32* timeData);

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

        /// Shift the zero frequency component to the centre of the spectrum.
        void _fftShift(Complex* spectrum, unsigned nChannels);

        /// Returns the sub-band ID range to be processed.
        void _threadProcessingIndices(unsigned& start, unsigned& end,
                unsigned nSubbands, unsigned nThreads, unsigned threadId);

        /// Set up processing buffers.
        unsigned _setupWorkBuffers(unsigned nSubbands,
                unsigned nPolariations, unsigned nChannels, unsigned nTaps);

    private:
        bool _buffersInitialised;
        unsigned _nChannels;
        unsigned _nThreads;

        PolyphaseCoefficients _ppfCoeffs;
        std::vector<float> _coeffs;

        unsigned _iOldestSamples; // Pointer to the oldest samples.

        fftwf_plan _fftPlan;

        // Work Buffers (need to have a buffer per thread).
        std::vector<std::vector<Complex> > _workBuffer;
        std::vector<std::vector<Complex> > _filteredData;
};

// Declare this class as a pelican module.
PELICAN_DECLARE_MODULE(PPFChanneliser)

}// namespace lofar
}// namespace pelican

#endif // PPF_CHANNELISER_H_
