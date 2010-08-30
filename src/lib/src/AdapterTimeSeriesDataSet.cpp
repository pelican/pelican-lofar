#include "AdapterTimeSeriesDataSet.h"

#include "LofarTypes.h"
#include "TimeSeriesDataSet.h"

#include "pelican/utility/ConfigNode.h"
#include "pelican/core/AbstractStreamAdapter.h"

#include <QtCore/QString>

#include <boost/cstdint.hpp>
#include <cmath>
#include <iostream>
#include <complex>
#include <vector>

using std::cout;
using std::endl;

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
AdapterTimeSeriesDataSet::AdapterTimeSeriesDataSet(const ConfigNode& config)
:AbstractStreamAdapter(config)
{
    // Grab configuration for the adapter
    _fixedPacketSize = config.getOption("fixedSizePackets", "value", "true").
            toLower().startsWith("true") ? true : false;
    _sampleBits = config.getOption("dataBitSize", "value", "0").toUInt();
    _nUDPPacketsPerChunk = config.getOption("udpPacketsPerIteration", "value", "0").toUInt();
    _nSamplesPerPacket = config.getOption("samplesPerPacket", "value", "0").toUInt();
    _nSamplesPerTimeBlock = config.getOption("outputChannelsPerSubband", "value", "0").toUInt();
    _nSubbands = config.getOption("subbandsPerPacket", "value", "0").toUInt();
    _nPolarisations = config.getOption("nRawPolarisations", "value", "0").toUInt();
    _clock = config.getOption("clock", "value", "200").toUInt();

    // Packet size variables.
    _packetSize = sizeof(UDPPacket);
    _headerSize = sizeof(UDPPacket::Header);

    // Must divide by 4 because we're using sampleBits * 2 for each value (complex data).
    _packetDataSize = _nSubbands * _nPolarisations * _nSamplesPerPacket * _sampleBits / 4;
    _dataSize = _fixedPacketSize ? 8130 : _packetDataSize;
    _paddingSize = _fixedPacketSize ? _packetSize - _headerSize - _dataSize : 0;

    // Temporary arrays for buffering data from the IO Device.
    _headerTemp.resize(_headerSize);
    _dataTemp.resize(_dataSize);
    _paddingTemp.resize(_paddingSize + 1);
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
void AdapterTimeSeriesDataSet::deserialise(QIODevice* in)
{
    // Sanity check on data blob dimensions and chunk size.
    _checkData();

    // UDP packet header.
    UDPPacket::Header header;

    char* headerTemp = &_headerTemp[0];
    char* dataTemp = &_dataTemp[0];
    char* paddingTemp = &_paddingTemp[0];

    // Loop over UDP packets
    for (unsigned p = 0u; p < _nUDPPacketsPerChunk; ++p) {

        // Read the header from the IO device.
        in->read(headerTemp, _headerSize);
        _readHeader(headerTemp, header);

        // First packet, extract time-stamp.
        if (p == 0u) {
            TYPES::TimeStamp timestamp;
            timestamp.setStationClockSpeed(_clock * 1000000);
            timestamp.setStamp (header.timestamp, header.blockSequenceNumber);
            _timeData->setLofarTimestamp(timestamp.itsTime);
            // Sample rate when condensed in chunk (ie. diff in time between chunks)
            _timeData->setBlockRate(_nSamplesPerTimeBlock);
        }

        // Read the useful data (depends on configured dimensions).
        in->read(dataTemp, _dataSize);
        _readData(p, dataTemp, _timeData);

        // Read off padding (from word alignment of the packet).
        in->read(paddingTemp, _paddingSize);
    }
}


/**
 * @details
 */
void AdapterTimeSeriesDataSet::_checkData()
{
    // Check for supported sample bits.
    if (_sampleBits != 8 && _sampleBits != 16)
        throw _err("Sample size (%1 bits) not supported.").arg(_sampleBits);

    // Check that there is something of to adapt.
    if (_chunkSize == 0) throw _err("Chunk size zero!");

    // Check the data blob passed to the adapter is allocated.
    if (!_data) throw _err("Cannot deserialise into an unallocated blob!.");

    unsigned nPacketSamples = _nSubbands * _nPolarisations * _nSamplesPerPacket;
    size_t usefulBits = nPacketSamples * _sampleBits * 2; // 2 for complex!
    size_t usefulBytes = usefulBits / 8;
    size_t packetSize = _fixedPacketSize ?
            _packetSize : _headerSize + usefulBytes;
    size_t udpDataBits = _fixedPacketSize ? 8130 * sizeof(char) * 8 : usefulBits;

    // Check the chunk size matches the expected number of UDPPackets.
    if (_chunkSize != packetSize * _nUDPPacketsPerChunk)
        throw _err("Chunk size '%1' != '%2' expected for %3 UDP packets.")
                .arg(_chunkSize).arg(packetSize).arg(_nUDPPacketsPerChunk);

    // Adapter dimensions must agree with packet data size.
    if (usefulBits > udpDataBits)
        throw _err("Dimensions decribe more data than fits into a UDP packet! "
                " (%1 > %2").arg(usefulBits).arg(udpDataBits);

    unsigned nTimesTotal = _nSamplesPerPacket * _nUDPPacketsPerChunk;
    if (nTimesTotal % _nSamplesPerTimeBlock != 0)
        throw _err("Number of time samples must be evenly divisible by the "
                "number of samples in a block.");

    // Resize the time stream data blob to match the adapter dimensions.
    unsigned nBlocks = nTimesTotal / _nSamplesPerTimeBlock;
    _timeData = (TimeSeriesDataSetC32*)_data;
    _timeData->resize(nBlocks, _nSubbands, _nPolarisations, _nSamplesPerTimeBlock);
}


/**
 * @details
 * Reads the UDP packet header from the IO device.
 *
 * @param[out] header	UDP packet header to be filled.
 * @param[in]  buffer	Char* buffer read from the IO device
 */
inline
void AdapterTimeSeriesDataSet::_readHeader(char* buffer, UDPPacket::Header& header)
{
    header = *reinterpret_cast<UDPPacket::Header*>(buffer);
//    _printHeader(header);
}


/**
 * @details
 * Reads the UDP data data section into the data blob data array.
 *
 * @param[in]  packet   Packet index to read data from.
 * @param[in]  buffer 	Char* buffer read from the IO device.
 * @param[out] data		time stream data data array (assumes double precision).
 */
void AdapterTimeSeriesDataSet::_readData(unsigned packet, char* buffer,
        TimeSeriesDataSetC32* data)
{
    unsigned tStart = packet * _nSamplesPerPacket;

    // Loop over dimensions in the packet and write into the data blob.
    unsigned iTimeBlock, index;
    Complex* times;
    unsigned iPtr = 0;
//    Real re, im;

    switch (_sampleBits)
    {
        case 8:
        {
            TYPES::i8complex i8c;
            for (unsigned s = 0; s < _nSubbands; ++s) {
                for (unsigned t = 0; t < _nSamplesPerPacket; ++t) {
                    iTimeBlock = (tStart + t) / _nSamplesPerTimeBlock;
                    for (unsigned p = 0; p < _nPolarisations; ++p) {
                        times = data->timeSeriesData(iTimeBlock, s, p);
                        index = tStart - (iTimeBlock * _nSamplesPerTimeBlock) + t;
                        i8c = *reinterpret_cast<TYPES::i8complex*>(&buffer[iPtr]);
                        times[index] = _makeComplex(i8c);
                        iPtr += sizeof(TYPES::i8complex);
                    }
                }
            }
            break;
        }
        case 16:
        {
            TYPES::i16complex i16c;
            size_t dataSize = sizeof(i16c);
            for (unsigned s = 0; s < _nSubbands; ++s) {
                for (unsigned t = 0; t < _nSamplesPerPacket; ++t) {

                    iTimeBlock = (tStart + t) / _nSamplesPerTimeBlock;

                    for (unsigned p = 0; p < _nPolarisations; ++p) {

                        times = data->timeSeriesData(iTimeBlock, s, p);

                        // Index into time vector at cube location (iTimeBlock, s, p)
                        index = tStart - (iTimeBlock * _nSamplesPerTimeBlock) + t;

                        i16c = *reinterpret_cast<TYPES::i16complex*>(&buffer[iPtr]);
//                        cout << "b = " << iTimeBlock << " s = " << s << " p = " << p << " index = " << index << endl;
                        times[index] = _makeComplex(i16c);
                        iPtr += dataSize;
                    }
                }
            }
            break;
        }
        default:
            throw _err("Bits per sample (%1) unsupported").arg(_sampleBits);
    };
}


/**
 * @details
 * Prints a udp packet header.
 *
 * @param header UDPPacket header for printing.
 */
void AdapterTimeSeriesDataSet::_printHeader(const UDPPacket::Header& header)
{
    cout << endl;
    cout << QString(80, '-').toStdString() << endl;
    cout << " UDPPacket::Header" << endl;
    cout << QString(80, '-').toStdString() << endl;
    cout << "* version             = " << (unsigned)header.version << endl;
    cout << "* sourceInfo          = " << (unsigned)header.sourceInfo << endl;
    cout << "* configuration       = " << (unsigned)header.configuration << endl;
    cout << "* station             = " << (unsigned)header.station << endl;
    cout << "* nrBeamlees          = " << (unsigned)header.nrBeamlets << endl;
    cout << "* nrBlocks            = " << (unsigned)header.nrBlocks << endl;
    cout << "* timestamp           = " << (unsigned)header.timestamp << endl;
    cout << "* blockSequenceNumber = " << (unsigned)header.blockSequenceNumber << endl;
    cout << QString(80, '-').toStdString() << endl;
    cout << endl;
}


inline AdapterTimeSeriesDataSet::Complex
AdapterTimeSeriesDataSet::_makeComplex(const TYPES::i8complex& z)
{
    return Complex( (Real) z.real(), (Real) z.imag() );
}


inline AdapterTimeSeriesDataSet::Complex
AdapterTimeSeriesDataSet::_makeComplex(const TYPES::i16complex& z)
{
    return Complex( (Real) z.real(), (Real) z.imag() );
}


inline QString AdapterTimeSeriesDataSet::_err(const QString& message)
{
    return QString("AdapterTimeSeriesDataSet: ") + message;
}

} // namespace lofar
} // namespace pelican
