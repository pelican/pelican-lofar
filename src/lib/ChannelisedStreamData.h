#ifndef CHANNELISEDSTREAMDATA_H
#define CHANNELISEDSTREAMDATA_H

/**
 * @file ChannelisedStreamData.h
 */

#include "pelican/data/DataBlob.h"
#include <vector>
#include <complex>
#include <QtCore/QIODevice>

namespace pelican {
namespace lofar {

/**
 * @class ChannelisedStreamData
 *
 * @brief
 * Container class to hold a buffer channelised (spectrum) stream data.
 *
 * @details
 * Populated by the channeliser module.
 */

template <class T>
class T_ChannelisedSteamData : public DataBlob
{
    public:
        /// Constructs an empty time channelised data blob.
        T_ChannelisedSteamData() : DataBlob() {
            _nSubbands = 0;
            _nPolarisations = 0;
            _nChannels = 0;
            _startFreq = 0.0;
            _channelFreqDelta = 0.0;
        }

        /// Constructs and assigns memory for a time stream buffer data blob.
        T_ChannelisedSteamData(const unsigned nSubbands,
                const unsigned nPolarisations, const unsigned nChannels)
        : DataBlob() {
            resize(nSubbands, nPolarisations, nChannels);
        }

        /// Destroys the time stream data blob.
        virtual ~T_ChannelisedSteamData() {}

    public:
        /// Clears the time stream data.
        void clear()
        {
            _data.clear();
            _nSubbands = 0;
            _nPolarisations = 0;
            _nChannels = 0;
            _startFreq = 0.0;
            _channelFreqDelta = 0.0;
        }

        /// Assign memory for the time stream data blob.
        void resize(const unsigned nSubbands, const unsigned nPolarisations,
                const unsigned nChannels)
        {
            _nSubbands = nSubbands;
            _nPolarisations = nPolarisations;
            _nChannels = nChannels;
            _data.resize(_nSubbands * _nPolarisations * _nChannels);
        }

        /// Returns the data index for a given sub-band, polarisation and
        /// sample.
        unsigned index(const unsigned subband, const unsigned polarisation,
                const unsigned channel)
        {
            return _nChannels * ( subband * _nPolarisations + polarisation) + channel;
        }

    public: // Accessor methods.
        /// Returns the number of entries in the data blob.
        unsigned size() const { return _data.size(); }

        /// Returns the number of sub-bands in the data.
        unsigned nSubbands() const { return _nSubbands; }

        /// Returns the number of polarisations in the data.
        unsigned nPolarisations() const { return _nPolarisations; }

        /// Returns the number of channels in the data.
        unsigned nChannels() const { return _nChannels; }

        /// Returns the start frequency of the spectrum.
        double startFrequency() const { return _startFreq; }

        /// Sets the start frequency of the spectrum.
        void setStartFrequency(const double value) { _startFreq = value; }

        /// Returns the channel frequency spacing.
        double channelFrequencyDelta() const { return _channelFreqDelta; }

        /// Sets the channel frequency spacing..
        void setChannelfrequencyDelta(const double value) {
            _channelFreqDelta = value;
        }

        /// Returns a pointer to the spectrum data.
        T* data() { return _data.size() > 0 ? &_data[0] : NULL; }

        /// Returns a pointer to the spectrum data (const overload).
        const T* data() const  { return _data.size() > 0 ? &_data[0] : NULL; }

        /// Returns a pointer to the spectrum data for the specified
        /// /p subband.
        T* data(const unsigned subband)
        {
            unsigned index =  subband * _nPolarisations * _nChannels;
            return (_data.size() > 0 && subband <= _nSubbands
                    && index < _data.size()) ? &_data[index] : NULL;
        }

        /// Returns a pointer to the spectrum data for the specified
        /// /p subband (const overload).
        const T* data(const unsigned subband) const
        {
            unsigned index = subband * _nPolarisations * _nChannels;
            return (_data.size() > 0 && subband < _nSubbands
                    && index < _data.size()) ? &_data[index] : NULL;
        }

        /// Returns a pointer to the spectrum data for the specified
        /// /p subband and /p polarisation.
        T* data(const unsigned subband, const unsigned polarisation)
        {
            unsigned index = _nChannels * (subband * _nPolarisations + polarisation);
            return (_data.size() > 0 && subband < _nSubbands
                    && polarisation < _nPolarisations && index < _data.size()) ?
                    &_data[index] : NULL;
        }

        /// Returns a pointer to the spectrum data for the specified
        /// /p sub-band (const overload).
        const T* data(const unsigned subband, const unsigned polarisation) const
        {
            unsigned index = 0;
            return (_data.size() > 0 && subband < _nSubbands
                    && polarisation < _nPolarisations && index < _data.size()) ?
                            &_data[index] : NULL;
        }

    protected:
        std::vector<T> _data;
        unsigned _nSubbands;
        unsigned _nPolarisations;
        unsigned _nChannels;
        double _startFreq;
        double _channelFreqDelta;
};


/**
 * @class ChannelisedStreaData
 *
 * @brief
 * Container class for double floating point format channelised data.
 *
 *
 * @details
 */
class ChannelisedStreamData : public T_ChannelisedSteamData<std::complex<double> >
{
    public:
        /// Constructs an empty time stream data blob.
        ChannelisedStreamData() : T_ChannelisedSteamData<std::complex<double> >() {}

        /// Constructs and assigns memory for a time stream buffer data blob.
        ChannelisedStreamData(const unsigned nSubbands, const unsigned nPolarisations,
                const unsigned nChannels)
        : T_ChannelisedSteamData<std::complex<double> >(nSubbands, nPolarisations,
                    nChannels) {}

        /// Constructs a data blob from a serial copy of the blob.
        ChannelisedStreamData(QIODevice& serialBlob)
        : T_ChannelisedSteamData<std::complex<double> >()
        {
            deserialise(serialBlob);
        }

        /// Destroys the time stream data blob.
        ~ChannelisedStreamData() {}

    public:
        /// Write the spectrum to file.
        void write(const QString& fileName) const;

        /// Serialises the data blob.
        void serialise(QIODevice&) const;

        /// Deserialises the data blob.
        void deserialise(QIODevice&);
};

}// namespace lofar
}// namespace pelican

#endif // CHANNELISEDSTREAMDATA_H
