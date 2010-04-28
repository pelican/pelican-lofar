#include "ChannelisedStreamData.h"


namespace pelican {
namespace lofar {

PELICAN_DECLARE_DATABLOB(ChannelisedStreamData)


/**
 * @details
 * Serialises the data blob.
 *
 * @return Returns a QByteArray containing the serialised data blob.
 */
QByteArray ChannelisedStreamData::serialise() const
{
    QByteArray serialData;

    // Header.
    // TODO: Add some sort of time stamp? (from the data blob base class.)
    serialData.append((char*)&_nSubbands, sizeof(unsigned));
    serialData.append((char*)&_nPolarisations, sizeof(unsigned));
    serialData.append((char*)&_nChannels, sizeof(unsigned));
    serialData.append((char*)&_startFreq, sizeof(double));
    serialData.append((char*)&_channelFreqDelta, sizeof(double));

    // Data.
    unsigned dataBytes = sizeof(std::complex<double>);
    serialData.append((char*)&_data[0], _data.size() * dataBytes);

    return serialData;
}


/**
 * @details
 * Deserialises the data blob.
 *
 * @note
 * The blob must have been serialised by the serailise method.
 *
 * @param[in] blob QByteArray containing a serialised version of the data blob.
 */
void ChannelisedStreamData::deserialise(const QByteArray& blob)
{
    // Read the header.
    unsigned offset = 0;
    const char* ptr = blob.data();
    _nSubbands = *reinterpret_cast<const unsigned*>(ptr + offset);
    offset = sizeof(unsigned);
    _nPolarisations = *reinterpret_cast<const unsigned*>(ptr + offset);
    offset += sizeof(unsigned);
    _nChannels = *reinterpret_cast<const unsigned*>(ptr + offset);
    offset += sizeof(unsigned);
    _startFreq = *reinterpret_cast<const double*>(ptr + offset);
    offset += sizeof(double);
    _channelFreqDelta = *reinterpret_cast<const double*>(ptr + offset);
    offset += sizeof(double);

    // Sanity check.
    unsigned headerSize = 3 * sizeof(unsigned) + 2 * sizeof(double);
    unsigned dataPoints = _nSubbands * _nPolarisations * _nChannels;
    unsigned dataSize = dataPoints * sizeof(std::complex<double>);
    if (headerSize + dataSize != blob.size()) {
        throw QString("ChannelisedStreamData:: Error deserialise data blob. "
                      "Size expected from header dosn't match blob size.");
    }

    // Resize the data blob to fit the byte array.
    resize(_nSubbands, _nPolarisations, _nChannels);

    // Read the data.
    std::complex<double>* data = &_data[0];
    for (unsigned i = 0; i < dataPoints; i++) {
        data[i] = *reinterpret_cast<const std::complex<double>* >(ptr + offset);
        offset += sizeof(std::complex<double>);
    }
}


} // namespace lofar
} // namespace pelican

