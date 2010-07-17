#include "AdapterSubbandTimeSeries.h"

#include "LofarTypes.h"
#include "SubbandTimeSeries.h"

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
 * Constructs a stream adapter for complex time stream data from a LOFAR station.
 *
 * Complex time stream data represents a time stream split into a number of
 * subbands.
 *
 * @param[in] config Pelican XML configuration node object.
 */
AdapterSubbandTimeSeries::AdapterSubbandTimeSeries(const ConfigNode& config)
:AbstractStreamAdapter(config)
{
    // Grab configuration for the adapter
    _fixedPacketSize = config.getOption("fixedSizePackets", "value", "true").
            toLower().startsWith("true") ? true : false;
    _sampleBits = config.getOption("sampleSize", "bits", "0").toUInt();
    _nUDPPacketsPerChunk = config.getOption("packetsPerChunk", "number", "0").toUInt();
    _nSamplesPerPacket = config.getOption("samplesPerPacket", "number", "0").toUInt();
    _nSamplesPerTimeBlock = config.getOption("samplesPerTimeBlock", "number", "0").toUInt();
    _nSubbands = config.getOption("subbands", "number", "0").toUInt();
    _nPolarisations = config.getOption("polarisations", "number", "0").toUInt();
}


/**
 * @details
 * Method to deserialise a single station sub-band time stream chunk.
 *
 * @param[in] in QIODevice containing a number of serialised UDP packets from
 * 				 the LOFAR RSP board.
 *
 * @note
 * There might be a problem with packing padding on different architectures
 * as the packet header + data section 4 byte word aligned and assumption is
 * made that any padding (if needed) is at the end of the packet.
 */
void AdapterSubbandTimeSeries::deserialise(QIODevice* in)
{
    // Sanity check on data blob dimensions and chunk size.
    _checkData();

    // Packet size variables.
    size_t packetSize = sizeof(UDPPacket);
    size_t headerSize = sizeof(UDPPacket::Header);

    // Must divide by 4 because we're using sampleBits * 2 for
    // each value (complex data).
    size_t packetDataSize = _nSubbands * _nPolarisations * _nSamplesPerPacket
            * _sampleBits / 4;
    size_t dataSize = _fixedPacketSize ? 8130 : packetDataSize;
    size_t paddingSize = _fixedPacketSize ? packetSize - headerSize - dataSize : 0;

    // Temporary arrays for buffering data from the IO Device.
    std::vector<char> headerTemp(headerSize);
    std::vector<char> dataTemp(dataSize);
    std::vector<char> paddingTemp(paddingSize + 1);

    // UDP packet header.
    UDPPacket::Header header;

    // TODO: Add time information to time data object (obtainable from seqid
    // and blockid using the TimeStamp code in IONproc).

    // Loop over UDP packets
    for (unsigned p = 0; p < _nUDPPacketsPerChunk; ++p) {

        // Read the header from the IO device.
        in->read(&headerTemp[0], headerSize);
        _readHeader(header, &headerTemp[0]);

        // TODO: copy timestamp stuff from AdapterTimeStream.

        // Read the useful data (depends on configured dimensions).
        in->read(&dataTemp[0], dataSize);
        _readData(_timeData, &dataTemp[0], p);

        // Read off padding (from word alignment of the packet).
        in->read(&paddingTemp[0], paddingSize);
    }
}


/**
 * @details
 */
void AdapterSubbandTimeSeries::_checkData()
{
    // Check for supported sample bits.
    if (_sampleBits != 8  && _sampleBits != 16) {
        throw QString("AdapterSubbandTimeSeries: Specified number of "
                "sample (%1) bits not supported.").arg(_sampleBits);
    }

    // Check that there is something of to adapt.
    if (_chunkSize == 0) {
        throw QString("AdapterSubbandTimeSeries: Chunk size Zero.");
    }

    unsigned packetUsefulBits = _nSubbands * _nPolarisations * _nSamplesPerPacket
            * _sampleBits * 2;
    unsigned packetSize = _fixedPacketSize ?
            sizeof(UDPPacket) : sizeof(UDPPacket::Header) + packetUsefulBits / 8;

    // Check the chunk size matches the expected number of UDPPackets.
    if (_chunkSize != packetSize * _nUDPPacketsPerChunk) {
        throw QString("AdapterSubbandTimeSeries: Chunk size '%1' doesn't "
                "match  the expected size '%2' for %3 UDP packets.")
                .arg(_chunkSize).arg(packetSize).arg(_nUDPPacketsPerChunk);
    }

    // Check the data blob passed to the adapter is allocated.
    if (!_data) {
        throw QString("AdapterSubbandTimeSeries: Cannot deserialise into an "
                      "unallocated blob!.");
    }

    // If any service data exists update the visibility dimensions from it.
    // FIXME: it exists all the time at the moment.
//    if (!_serviceData.empty()) {
//        _updateDimensions();
//    }

    // Check that the adapter dimensions agree with what could come from
    // packets. TODO test this...
    unsigned udpDataBits = _fixedPacketSize ?
            8130 * sizeof(char) * 8 : packetUsefulBits;
    if (packetUsefulBits > udpDataBits) {
        throw QString("AdapterSubbandTimeSeries: Adapter dimensions specify "
                "more data than fits into a UDP packet! (%1 > %2)")
                .arg(packetUsefulBits).arg(udpDataBits);
    }

    unsigned nTimesTotal = _nSamplesPerPacket * _nUDPPacketsPerChunk;

    if (nTimesTotal % _nSamplesPerTimeBlock != 0) {
        throw QString("AdapterSubbandTimeSeries:: Number of time samples not "
                "evenly devisable by the number of samples in a block.");
    }

    // Resize the time stream data blob being read into to match the adapter
    // dimensions.
    unsigned nTimeBlocks = nTimesTotal / _nSamplesPerTimeBlock;
    _timeData = static_cast<SubbandTimeSeriesC32*>(_data);
    _timeData->resize(nTimeBlocks, _nSubbands, _nPolarisations, _nSamplesPerTimeBlock);

//    std::cout << "AdapterSubbandTimeSeries::_checkData(): nTimeBlocks = "
//              << nTimeBlocks << std::endl;
//    std::cout << "AdapterSubbandTimeSeries::_checkData(): nSubbands = "
//                  << _nSubbands << std::endl;
//    std::cout << "AdapterSubbandTimeSeries::_checkData(): nPolarisations = "
//                  << _nPolarisations << std::endl;
//    std::cout << "AdapterSubbandTimeSeries::_checkData(): nSamplesPerTimeBlock = "
//                  << _nSamplesPerTimeBlock << std::endl;
}


/**
 * @details
 * Reads the UDP packet header from the IO device.
 *
 * @param[out] header	UDP packet header to be filled.
 * @param[in]  buffer	Char* buffer read from the IO device
 */
void AdapterSubbandTimeSeries::_readHeader(UDPPacket::Header& header,
        char* buffer)
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
void AdapterSubbandTimeSeries::_readData(SubbandTimeSeriesC32* timeSeries,
        char* buffer, unsigned packetIndex)
{
    typedef std::complex<float> fComplex;
    unsigned tStart = packetIndex * _nSamplesPerPacket; // TODO check this.

//    std::cout << "tStart = " << tStart << std::endl;
//    std::cout << "_nSamplesPerPacket = " << _nSamplesPerPacket << std::endl;
//    std::cout << "_nPolarisations = " << _nPolarisations << std::endl;

    // Loop over dimensions in the packet and write into the data blob.
    for (unsigned iPtr = 0, t = 0; t < _nSamplesPerPacket; ++t) {

        unsigned iTimeBlock = (tStart + t) / _nSamplesPerTimeBlock;
//        std::cout << "iTimeBlock = " << iTimeBlock << std::endl;

        for (unsigned c = 0; c < _nSubbands; ++c) {
            for (unsigned p = 0; p < _nPolarisations; ++p) {

                fComplex* data = timeSeries->ptr(iTimeBlock, c, p)->ptr();

                // TODO: This needs double checking...
                unsigned index = tStart - (iTimeBlock * _nSamplesPerTimeBlock) + t;


                if (_sampleBits == 8) {
                    TYPES::i8complex i8c = *reinterpret_cast<TYPES::i8complex*>(&buffer[iPtr]);

                    // TODO VITAL CONVERSION OF INT8 to float!!! (see lofar code)

                    data[index].real() = float(i8c.real());
                    data[index].imag() = float(i8c.imag());
                    iPtr += sizeof(TYPES::i8complex);
                }
                else if (_sampleBits == 16) {
                    TYPES::i16complex i16c = *reinterpret_cast<TYPES::i16complex*>(&buffer[iPtr]);
                    data[index].real() = float(i16c.real());
                    data[index].imag() = float(i16c.imag());
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
void AdapterSubbandTimeSeries::_updateDimensions()
{
    // TODO: FIX this !!!
//	throw QString("AdapterSubbandTimeSeries::_updateDimensions(): "
//            "Updating dimensions from service data not currently supported.");
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
void AdapterSubbandTimeSeries::_printHeader(const UDPPacket::Header& header)
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
