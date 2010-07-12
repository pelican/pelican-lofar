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

class SubbandTimeStreamC32;
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
 * 			<coefficients fileName="coeffs.dat" nTaps="8"/>
 * 		</ChanneliserPolyphase>
 * \verbatim
 *
 * - channels: Number of channels generated in the spectra.
 * - processingThreads: Number of threads to parallelise over.
 * - coefficients:
 * 		 fileName: file containing the PPF coefficients.
 * 		 nTaps: Number of filter taps in the PPF coefficient data
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
        void run(const SubbandTimeStreamC32* timeSeries,
                SubbandSpectraC32* spectra);

    private:
        /// Sainity checking.
        void _checkData(const SubbandTimeStreamC32* timeData);

        /// Update the sample buffer.
        void _updateBuffer(const Complex* samples, unsigned nSamples,
                Complex* buffer, unsigned bufferSize);

        /// Filter the matrix of samples (dimensions nTaps by nChannels)
        /// to create a vector of samples for the FFT.
        void _filter(const Complex* sampleBuffer, unsigned nTaps,
                unsigned nChannels, const double* coeffs,
                Complex* filteredSamples);

        /// FFT filtered samples to form a spectrum.
        void _fft(const Complex* samples, unsigned nSamples,
                Complex* spectrum);

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
        PolyphaseCoefficients _coeffs;
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
