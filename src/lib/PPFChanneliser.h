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

class TimeSeriesC32;
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

    public:
        /// Constructs the channeliser module.
        PPFChanneliser(const ConfigNode& config);

        /// Destroys the channeliser module.
        ~PPFChanneliser();

        /// Method converting the time stream to a spectrum.
        void run(const TimeSeriesC32* timeSeries, SubbandSpectraC32* spectra);

    private:
        /// Sainity checking.
        void _checkData(const TimeSeriesC32* timeData);

        /// Update the sample buffer.
        void _updateBuffer(const complex<float>* samples,
                unsigned nSamples, complex<float>* buffer,
                unsigned bufferSize);

        /// Filter the matrix of samples (dimensions nTaps by nChannels)
        /// to create a vector of samples for the FFT.
        void _filter(const complex<float>* sampleBuffer,
                unsigned nTaps,  unsigned nChannels,
                const double* coeffs, complex<float>* filteredSamples);

        /// FFT filtered samples to form a spectrum.
        void _fft(const complex<float>* samples, unsigned nSamples,
                complex<float>* spectrum);

        /// Shift the zero frequency component to the centre of the spectrum.
        void _fftShift(complex<float>* spectrum, unsigned nChannels);

        /// Returns the sub-band ID range to be processed.
        void _threadSubbandRange(unsigned& start, unsigned& end,
                unsigned nSubbands, unsigned nThreads, unsigned threadId);

        /// Set up processing buffers.
        unsigned _setupBuffers(unsigned nSubbands, unsigned nChannels,
                unsigned nFilterTaps);

    private:
        bool _buffersInitialised;
        unsigned _nChannels;

        // Work Buffers (need to have a buffer per thread).
        std::vector<std::vector<complex<float> > > _subbandBuffer;
        std::vector<std::vector<complex<float> > > _filteredData;

        PolyphaseCoefficients _coeffs;

        unsigned _nThreads;

        fftwf_plan _fftPlan;
};

// Declare this class as a pelican module.
PELICAN_DECLARE_MODULE(PPFChanneliser)

}// namespace lofar
}// namespace pelican

#endif // PPF_CHANNELISER_H_
