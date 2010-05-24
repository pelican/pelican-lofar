#include "ChannelisedStreamData.h"
#include <QtCore/QIODevice>

#include <iostream>

namespace pelican {
namespace lofar {

PELICAN_DECLARE_DATABLOB(ChannelisedStreamData)

/**
 * @details
 * Serialises the data blob.
 */
void ChannelisedStreamData::serialise(QIODevice& out) const
{
    // Header.
    // TODO: Add some sort of time stamp? (from the data blob base class.)
    out.write((char*)&_nSubbands, sizeof(unsigned));
    out.write((char*)&_nPolarisations, sizeof(unsigned));
    out.write((char*)&_nChannels, sizeof(unsigned));
    out.write((char*)&_startFreq, sizeof(double));
    out.write((char*)&_channelFreqDelta, sizeof(double));

    // Data.
    out.write((char*)&_data[0], _data.size() * sizeof(std::complex<double>));
}


/**
 * @details
 * Deserialises the data blob.
 */
void ChannelisedStreamData::deserialise(QIODevice& in)
{
    // Read the header.
    unsigned offset = 0;
    in.read((char*)&_nSubbands, sizeof(unsigned));
    in.read((char*)&_nPolarisations, sizeof(unsigned));
    in.read((char*)&_nChannels, sizeof(unsigned));
    in.read((char*)&_startFreq, sizeof(double));
    in.read((char*)&_channelFreqDelta, sizeof(double));

    unsigned dataPoints = _nSubbands * _nPolarisations * _nChannels;
    unsigned dataSize = dataPoints * sizeof(std::complex<double>);

    // Resize the data blob to fit the byte array.
    resize(_nSubbands, _nPolarisations, _nChannels);

    // Read the data.
    in.read((char*)&_data[0], dataSize);
}

	} // namespace lofar
} // namespace pelican

