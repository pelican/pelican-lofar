#include "AdapterTimeStream.h"

#include "pelican/data/TimeStreamData.h"
#include "pelican/utility/ConfigNode.h"
#include "LofarTypes.h"
#include <cmath>
#include <boost/cstdint.hpp>
#include <QString>

namespace pelican {

namespace lofar {

PELICAN_DECLARE_ADAPTER(AdapterTimeStream)


/**
 * @details
 * Constructs a stream adapter for time stream data from a LOFAR station.
 *
 * @param[in] config Pelican XML configuration node object.
 */
AdapterTimeStream::AdapterTimeStream(const ConfigNode& config)
:AbstractStreamAdapter(config)
{
    // Grab configuration for the adapter
    _nUDPPackets = config.getOption("packetsPerChunk", "number", "0").toUInt();
    _nSubbands = config.getOption("subbands", "number", "0").toUInt();
    _nPolarisations = config.getOption("polarisations", "number", "0").toUInt();
    _nSamples = config.getOption("samples", "number", "0").toUInt();
    _sampleBits = config.getOption("sampleSize", "bits", "0").toUInt();
}


/**
 * @details
 * Method to deserialise a single station time stream chunk.
 *
 * @param[in] in QIODevice containing a serialised version of a LOFAR
 *               visibility data set.
 *
 * @note
 * There might be a problem with packing padding on different architectures
 * as the packet header + data section 4 byte word aligned and assumption is
 * made that any padding (if needed) is at the end of the packet.
 */
void AdapterTimeStream::deserialise(QIODevice* in)
{
    // Sanity check on data blob dimensions and chunk size.
    _checkData();

    // Packet size variables.
    size_t packetSize = sizeof(UDPPacket);
    size_t headerSize = sizeof(UDPPacket::Header);
    size_t dataSize = 8130;
    size_t paddingSize = packetSize - headerSize - dataSize;

    // Temporary arrays for buffering data from the IO Device.
    std::vector<char> headerTemp(headerSize);
    std::vector<char> dataTemp(dataSize);
    std::vector<char> paddingTemp(paddingSize);

    // Data blob to read into.
    std::complex<double>* data = _timeData->data();

    // UDP packet header.
    UDPPacket::Header header;

    // Loop over UDP packets
    for (unsigned p = 0; p < _nUDPPackets; ++p) {

        // Read the header from the IO device.
        in->read(&headerTemp[0], headerSize);
        _readHeader(header, &headerTemp[0]);

        // Read the useful data (depends on configured dimensions).
        in->read(&dataTemp[0], dataSize);
        _readData(data, &dataTemp[0]);

        // Read off padding (from word alignment of the packet).
        in->read(&paddingTemp[0], paddingSize);
    }
}


/**
 * @details
 */
void AdapterTimeStream::_checkData()
{
    // TODO: throw on unsupported sample bits.
    //

    // Check that there is something of to adapt.
    if (_chunkSize == 0) {
        throw QString("AdapterTimeStream: Chunk size Zero.");
    }

    unsigned packetSize = sizeof(UDPPacket);

    // Check the chunk size matches the expected number of UDPPackets
    if (_chunkSize != packetSize * _nUDPPackets) {
        throw QString("AdapterTimeStream: Chunk size '%1' dosnt match the expected "
                      " size '%2' for %3 UDP packets.").
                      arg(_chunkSize).arg(packetSize).arg(_nUDPPackets);
    }

    // Check the data blob passed to the adapter is allocated.
    if (_data == NULL) {
        throw QString("AdapterTimeStream: Cannot deserialise into an "
                      "unallocated blob!.");
    }

    // If any service data exists update the visibility dimensions from it.
    if (!_serviceData.empty()) {
        _updateDimensions();
    }

    // Check that the adapter dimensions agree with what could come from
    // packets.
    unsigned packetBits = _nSubbands * _nPolarisations * _nSamples * _sampleBits;
    unsigned udpDataBits = 8130 * 8;
    if (packetBits > udpDataBits) {
        throw QString("AdapterTimeStream: Adapter dimensions specify more data "
                      "than fits into a UDP packet! (%1 > %2)")
                      .arg(packetBits).arg(udpDataBits);
    }

    // Resize the time stream data blob being read into to match the adapter
    // dimensions.
    _timeData = static_cast<TimeStreamData*>(_data);
    _timeData->resize(_nSubbands, _nPolarisations, _nSamples);
}


/**
 * @details
 * Reads the UDP packet header from the IO device.
 *
 * @param header UDP packet header to be filled.
 * @param buffer Char* buffer read from the IO device
 */
void AdapterTimeStream::_readHeader(UDPPacket::Header& header, char* buffer)
{
    header.version = *reinterpret_cast<uint8_t*>(&buffer[0]);
    header.sourceInfo = *reinterpret_cast<uint8_t*>(&buffer[1]);
    header.configuration = *reinterpret_cast<uint16_t*>(&buffer[2]);
    header.station = *reinterpret_cast<uint16_t*>(&buffer[4]);
    header.nrBeamlets = *reinterpret_cast<uint8_t*>(&buffer[6]);
    header.nrBlocks = *reinterpret_cast<uint8_t*>(&buffer[7]);
    header.timestamp = *reinterpret_cast<uint32_t*>(&buffer[8]);
    header.blockSequenceNumber = *reinterpret_cast<uint32_t*>(&buffer[12]);
//    _printHeader(header);
}


/**
 * @details
 * Reads the udp data data section into the data blob data array.
 *
 * @param data time stream data data array (assumes double float format).
 * @param buffer Char* buffer read from the IO device.
 */
void AdapterTimeStream::_readData(std::complex<double>* data, char* buffer)
{
    // Types (TODO use the ones defined elsewhere...)
    typedef std::complex<double> fComplex64;
    typedef TYPES::i4complex iComplex8;
    typedef TYPES::i8complex iComplex16;
    typedef TYPES::i16complex iComplex32;

    // Loop over dimensions in the packet and write into the data blob.
    for (unsigned iPtr = 0, t = 0; t < _nSamples; ++t) {
        for (unsigned c = 0; c < _nSubbands; ++c) {
            for (unsigned p = 0; p < _nPolarisations; ++p) {

                unsigned packetIndex = _nPolarisations * (t * _nSubbands + c) + p;;
                unsigned blobIndex = _nSamples * (c * _nPolarisations + p) + t;

                if (_sampleBits == 4) {
                    iComplex8 value = *reinterpret_cast<iComplex8*>(&buffer[iPtr]);
                    data[blobIndex] = fComplex64(double(value.real()), double(value.imag()));
                    iPtr += sizeof(iComplex8);
                }
                else if (_sampleBits == 8) {
                    iComplex16 i8c = *reinterpret_cast<iComplex16*>(&buffer[iPtr]);
                    data[blobIndex] = fComplex64(double(i8c.real()), double(i8c.imag()));
                    iPtr += sizeof(iComplex16);
                }
                else if (_sampleBits ==16) {
                    iComplex32 i16c = *reinterpret_cast<iComplex32*>(&buffer[iPtr]);
                    data[blobIndex] = fComplex64(double(i16c.real()), double(i16c.imag()));
                    iPtr += sizeof(iComplex32);
                }
                else {
                    throw QString("AdapterTimeStream: Specified number of bits "
                            " per sample (%1) unsupported").arg(_sampleBits);
                }

//                std::cout << data[blobIndex] << std::endl;
            }
        }
    }
}


/**
 * @details
 * Updates the time stream data dimensions from the service data passed
 * down from the adapter configuration.
 */
void AdapterTimeStream::_updateDimensions()
{
    // Example (if any service data exists):
    // _nTimes = serverData.nTimes;
    // _timeData.resize(_nTimes);
}


/**
 * @details
 * Prints a udp packet header.
 *
 * @param header UDPPacket header for printing.
 */
void AdapterTimeStream::_printHeader(const UDPPacket::Header& header)
{
    std::cout << std::endl;
    std::cout << QString(80, '-').toStdString() << std::endl;
    std::cout << " UDPPacket::Header" << std::endl;
    std::cout << QString(80, '-').toStdString() << std::endl;
    std::cout << "* version             = " << unsigned(header.version) << std::endl;
    std::cout << "* sourceInfo          = " << unsigned(header.sourceInfo) << std::endl;
    std::cout << "* configuration       = " << unsigned(header.configuration) << std::endl;
    std::cout << "* station             = " << unsigned(header.station) << std::endl;
    std::cout << "* nrBeamlees          = " << unsigned(header.nrBeamlets) << std::endl;
    std::cout << "* nrBlocks            = " << unsigned(header.nrBlocks) << std::endl;
    std::cout << "* timestamp           = " << unsigned(header.timestamp) << std::endl;
    std::cout << "* blockSequenceNumber = " << unsigned(header.blockSequenceNumber) << std::endl;
    std::cout << QString(80, '-').toStdString() << std::endl;
    std::cout << std::endl;
}


} // namespace lofar
} // namespace pelican
