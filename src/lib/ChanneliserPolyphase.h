#ifndef CHANNELISERPOLYPHASE_H
#define CHANNELISERPOLYPHASE_H

/**
 * @file ChanneliserPolyphase.h
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

class TimeStreamData;
class ChannelisedStreamData;

/**
 * @class ChanneliserPolyphase
 *
 * @ingroup pelican_lofar
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
 * 		</ChanneliserPolyphase>
 * \verbatim
 *
 * - channels: Number of channels to produce.
 *
 */

class ChanneliserPolyphase : public AbstractModule
{
	private:
		friend class ChanneliserPolyphaseTest;

	public:
        /// Constructs the channeliser module.
        ChanneliserPolyphase(const ConfigNode& config);

        /// Destroys the channeliser module.
        ~ChanneliserPolyphase();

        /// Method converting the time stream to a spectrum.
        void run(const TimeStreamData* timeData,
        		const PolyphaseCoefficients* filterCoeff,
        		ChannelisedStreamData* spectra);

	private:
        /// Sainity checking.
        void _checkData(const TimeStreamData* timeData,
        		const PolyphaseCoefficients* filterCoeff);

        /// Update the sample buffer.
        void _updateBuffer(const complex<double>* samples,
        		unsigned nSamples, complex<double>* buffer,
        		unsigned bufferSize);

        /// Filter the matrix of samples (dimensions nTaps by nChannels)
        /// to create a vector of samples for the FFT.
        void _filter(const complex<double>* sampleBuffer,
        		unsigned nTaps,  unsigned nChannels,
        		const double* coefficients, complex<double>* filteredSamples);

        /// FFT filtered samples to form a spectrum.
        void _fft(const complex<double>* samples, unsigned nSamples,
        		complex<double>* spectrum);

        /// Shift the zero frequency component to the centre of the spectrum.
        void _fftShift(complex<double>* spectrum, unsigned nChannels);

        /// Returns the sub-band ID range to be processed.
        void _threadSubbandRange(unsigned& start, unsigned& end,
        		unsigned nSubbands, unsigned nThreads, unsigned threadId);

        /// Set up processing buffers.
        unsigned _setupBuffers( unsigned nSubbands, unsigned nChannels,
        		unsigned nFilterTaps);

    private:
        bool _buffersInitialised; ///< Flag set if the buffers have been initialised.
        unsigned _nChannels;	  ///< Number of channels to produce per subband.

        // TODO: The following should probably be matrixes...
        std::vector<std::vector<complex<double> > > _subbandBuffer;
        std::vector<std::vector<complex<double> > > _filteredData;

        unsigned _nThreads;

        fftw_plan _fftPlan;
};

// Declare this class as a pelican module.
PELICAN_DECLARE_MODULE(ChanneliserPolyphase)

}// namespace lofar
}// namespace pelican

#endif // TIMESTREAMDATA_H_
