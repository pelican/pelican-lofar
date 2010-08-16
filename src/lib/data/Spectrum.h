#ifndef SPECTRUM_H_
#define SPECTRUM_H_

/**
 * @file Spectrum.h
 */

#include "pelican/data/DataBlob.h"

#include <vector>
#include <complex>

namespace pelican {
namespace lofar {

/**
 * @class T_Spectrum
 *
 * @brief
 * Container class to hold a spectrum.
 *
 * @details
 */

template <class T> class Spectrum
{
    public:
        /// Constructs an empty spectrum.
        Spectrum() : _startFreq(0.0), _frequencyIncrement(0.0) {}

        /// Constructs and assigns memory for the spectrum.
        Spectrum(unsigned nChannels)
        : _startFreq(0.0), _frequencyIncrement(0.0)
        {
            resize(nChannels);
        }

        /// Destroys the spectrum.
        virtual ~Spectrum() {}

    public:
        /// Clears the time stream data.
        void clear()
        {
            _channels.clear();
            _startFreq = 0.0;
            _frequencyIncrement = 0.0;
        }

        /// Resize the spectrum to the number of channels specified.
        void resize(unsigned nChannels) { _channels.resize(nChannels); }

    public:
        /// Returns the number of channels in the spectrum.
        unsigned nChannels() const { return _channels.size(); }

        /// Returns the start frequency of the spectrum.
        double startFrequency() const { return _startFreq; }

        /// Sets the start frequency of the spectrum.
        void setStartFrequency(double value) { _startFreq = value; }

        /// Returns the channel frequency spacing.
        double frequencyIncrement() const { return _frequencyIncrement; }

        /// Sets the channel frequency spacing..
        void setFrequencyIncrement(double value) {
            _frequencyIncrement = value;
        }

        /// Returns a pointer to the spectrum data.
        T* getData() { return _channels.size() > 0 ? &_channels[0] : NULL; }

        /// Returns a pointer to the spectrum data (const overload).
        const T* getData() const {
            return _channels.size() > 0 ? &_channels[0] : NULL;
        }

        /// To be deprecated soon (dont use)
        T* ptr() { return _channels.size() > 0 ? &_channels[0] : NULL; }

        /// To be deprecated soon (dont use)
        const T* ptr() const {
            return _channels.size() > 0 ? &_channels[0] : NULL;
        }

    protected:
        std::vector<T> _channels;
        double _startFreq;
        double _frequencyIncrement;
};


}// namespace lofar
}// namespace pelican

#endif // SPECTRUM_H_
