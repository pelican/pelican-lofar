#ifndef CHANNELISERPOLYPHASE_H
#define CHANNELISERPOLYPHASE_H

/**
 * @file ChanneliserPolyphase.h
 */

#include "pelican/modules/AbstractModule.h"
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
 * @brief
 * Module to channelise a time stream data blob.
 *
 * @details
 * Channelises time stream data using a polyphase channelising filter.
 */

class ChanneliserPolyphase : public AbstractModule
{
	public:
        /// Constructs the channeliser module.
        ChanneliserPolyphase(const ConfigNode& config);

        /// Destroys the channeliser module.
        ~ChanneliserPolyphase() {
            fftw_destroy_plan(_fftPlan);
        }

        /// Method converting the time stream to a spectrum.
        void run(const TimeStreamData* timeData, ChannelisedStreamData* spectrum);

    private:
        /// Update the sample buffer.
        void _updateBuffer(const complex<double>* samples,
        		const unsigned nSamples, complex<double>* buffer,
        		const unsigned bufferSize);

        /// Filter the matrix of samples (dimensions nTaps by nChannels)
        /// to create a vector of samples for the FFT.
        void _filter(const complex<double>* sampleBuffer,
        		const unsigned nTaps, const unsigned nChannels,
        		const complex<double>* coefficients,
        		complex<double>* filteredSamples);

        /// FFT filtered samples to form a spectrum.
        void _fft(const complex<double>* samples, const unsigned nSamples,
        		complex<double>* spectrum);

    private:
        unsigned _nChannels;
        unsigned _nFilterTaps;
        unsigned _nSubbands;
        ChannelisedStreamData* _spectrum;
        std::vector<std::complex<double> > _filterCoeff;
        std::vector<std::vector<complex<double> > > _subbandBuffer;
        std::vector<std::vector<std::complex<double> > > _filteredBuffer;

        fftw_plan _fftPlan;
        fftw_complex* _fftwIn;
        fftw_complex* _fftwOut;
};


}// namespace lofar
}// namespace pelican

#endif // TIMESTREAMDATA_H_
