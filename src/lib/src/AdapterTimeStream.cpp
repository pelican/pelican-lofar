#include "AdapterTimeStream.h"

#include "LofarTypes.h"
#include "TimeStreamData.h"

#include "pelican/utility/ConfigNode.h"
#include "pelican/core/AbstractStreamAdapter.h"

#include <QtCore/QString>

#include <boost/cstdint.hpp>
#include <cmath>
#include <iostream>

namespace pelican {
namespace lofar {

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
    _nSamples = config.getOption("samplesPerPacket", "number", "0").toUInt();
    _sampleBits = config.getOption("sampleSize", "bits", "0").toUInt();
    _clock = config.getOption("clockSpeed", "value", "200").toUInt();
    _fixedPacketSize = config.getOption("fixedSizePackets", "value", "true").
    		toLower().startsWith("true") ? true : false;
    _combinePols = config.getOption("combinePolarisations", "value", "false").
        		toLower().startsWith("true") ? true : false;
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

    // Must divide by 4 because we're using sampleBits * 2 for
    // each value (complex data).
    size_t dataSize = _fixedPacketSize ?
    		8130 : _nSubbands * _nPolarisations * _nSamples * _sampleBits / 4;
    size_t paddingSize = _fixedPacketSize ? packetSize - headerSize - dataSize : 0;

    // Temporary arrays for buffering data from the IO Device.
    std::vector<char> headerTemp(headerSize);
    std::vector<char> dataTemp(dataSize);
    std::vector<char> paddingTemp(paddingSize + 1);

    // Data blob to read into.
    std::complex<double>* data = _timeData->data();

    // UDP packet header.
    UDPPacket::Header header;

    // Loop over UDP packets
    for (unsigned p = 0; p < _nUDPPackets; ++p) {

        // Read the header from the IO device.
        in->read(&headerTemp[0], headerSize);
        _readHeader(header, &headerTemp[0]);
 
        // First packet, extract timestamp
        if (p == 0) {
            TYPES::TimeStamp timestamp;
            timestamp.setStationClockSpeed(_clock * 1000000);
            timestamp.setStamp (header.timestamp, header.blockSequenceNumber);
            _timeData -> setLofarTimestamp(timestamp.itsTime);
            _timeData -> setBlockRate(_nSamples * _nUDPPackets); // sample rate when condensed in chunk (ie. diff in time between chunks)
        }

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
    // Check for supported sample bits.
	if (_sampleBits != 4 && _sampleBits != 8  && _sampleBits != 16) {
        throw QString("AdapterTimeStream: Specified number of sample bits not "
                      "supported.");
    }

    // Check that there is something of to adapt.
    if (_chunkSize == 0) {
        throw QString("AdapterTimeStream: Chunk size Zero.");
    }

    unsigned packetBits = _nSubbands * _nPolarisations * _nSamples * _sampleBits * 2;
    unsigned packetSize = _fixedPacketSize ?
    		sizeof(UDPPacket) : sizeof(UDPPacket::Header) + packetBits / 8;

    // Check the chunk size matches the expected number of UDPPackets
    if (_chunkSize != packetSize * _nUDPPackets) {
        throw QString("AdapterTimeStream: Chunk size '%1' doesn't match the expected "
                      " size '%2' for %3 UDP packets.").
                      arg(_chunkSize).arg(packetSize).arg(_nUDPPackets);
    }

    // Check the data blob passed to the adapter is allocated.
    if (!_data) {
        throw QString("AdapterTimeStream: Cannot deserialise into an "
                      "unallocated blob!.");
    }

    // If any service data exists update the visibility dimensions from it.
    if (!_serviceData.empty()) {
        _updateDimensions();
    }

    // Check that the adapter dimensions agree with what could come from
    // packets. TODO test this...
    unsigned udpDataBits = _fixedPacketSize ? 8130 * sizeof(char) * 8 : packetBits;
    if (packetBits > udpDataBits) {
        throw QString("AdapterTimeStream: Adapter dimensions specify more data "
                      "than fits into a UDP packet! (%1 > %2)")
                      .arg(packetBits).arg(udpDataBits);
    }

    // Resize the time stream data blob being read into to match the adapter
    // dimensions.
    _timeData = static_cast<TimeStreamData*>(_data);
    _timeData->resize(_nSubbands, _nPolarisations, _nSamples * _nUDPPackets);
}


/**
 * @details
 * Reads the UDP packet header from the IO device.
 *
 * @param[out] header	UDP packet header to be filled.
 * @param[in]  buffer	Char* buffer read from the IO device
 */
void AdapterTimeStream::_readHeader(UDPPacket::Header& header, char* buffer)
{
    header = *reinterpret_cast<UDPPacket::Header*>(buffer);
//    _printHeader(header);
}


/**
 * @details
 * Reads the udp data data section into the data blob data array.
 *
 * @param[out] data		time stream data data array (assumes double precision).
 * @param[in]  buffer 	Char* buffer read from the IO device.
 */
void AdapterTimeStream::_readData(std::complex<double>* data, char* buffer)
{
    typedef std::complex<double> dComplex;

    // Loop over dimensions in the packet and write into the data blob.
    // TODO: Check endiannes
    for (unsigned iPtr = 0, t = 0; t < _nSamples; ++t) {
        for (unsigned c = 0; c < _nSubbands; ++c) {
            for (unsigned p = 0; p < _nPolarisations; ++p) {

                unsigned packetIndex = _nPolarisations * (t * _nSubbands + c) + p;;
                unsigned blobIndex = _nSamples * (c * _nPolarisations + p) + t;

                if (_sampleBits == 4) {
                    TYPES::i4complex value = *reinterpret_cast<TYPES::i4complex*>(&buffer[iPtr]);
                    data[blobIndex] = dComplex(double(value.real()), double(value.imag()));
                    iPtr += sizeof(TYPES::i4complex);
                }
                else if (_sampleBits == 8) {
                    TYPES::i8complex i8c = *reinterpret_cast<TYPES::i8complex*>(&buffer[iPtr]);
                    data[blobIndex] = dComplex(double(i8c.real()), double(i8c.imag()));
                    iPtr += sizeof(TYPES::i8complex);
                }
                else if (_sampleBits ==16) {
                    TYPES::i16complex i16c = *reinterpret_cast<TYPES::i16complex*>(&buffer[iPtr]);
                    data[blobIndex] = dComplex(double(i16c.real()), double(i16c.imag()));
                    iPtr += sizeof(TYPES::i16complex);
                }
                else {
                    throw QString("AdapterTimeStream: Specified number of bits "
                            " per sample (%1) unsupported").arg(_sampleBits);
                }
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


/**
 * @details
 * Combine polarisations.
 *
 * @param in
 * @param nSubbands
 * @param nPolarisations
 * @param nSamples
 * @param out
 */
void AdapterTimeStream::_combinePolarisations(std::complex<double>* in,
		const unsigned nSubbands, const unsigned nPolarisations,
		const unsigned nSamples, std::complex<double>* out)
{
	for (unsigned index = 0, c = 0; c < nSubbands; ++c) {
		for (unsigned t = 0; t < nSamples; ++t) {
			unsigned iPol1 =  nSamples * (c * nPolarisations + 0) + t;
			unsigned iPol2 =  nSamples * (c * nPolarisations + 1) + t;
			out[index] = in[iPol1] + in[iPol2]; // TODO: combine properly.
			index++;
		}
	}
}


} // namespace lofar
} // namespace pelican
