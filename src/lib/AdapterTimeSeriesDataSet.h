#ifndef ADAPTER_TIME_SERIES_DATA_SET_H
#define ADAPTER_TIME_SERIES_DATA_SET_H

#include "pelican/core/AbstractStreamAdapter.h"
#include "LofarUdpHeader.h"
#include "LofarTypes.h"
#include <complex>

#include "timer.h"

/**
 * @file AdapterTimeSeriesDataSet.h
 */

extern TimerData adapterTime;

namespace pelican {

class ConfigNode;

namespace lofar {

class TimeSeriesDataSetC32;

/**
 * @class AdapterTimeSeriesDataSet
 *
 * @ingroup pelican_lofar
 *
 * @brief
 * Adapter to deserialise chunks of UDP packets from a LOFAR station RSP board.
 *
 * @details
 * Adapter to deserialise chunks of UDP packets from a LOFAR station RSP board.
 *
 * \section Configuration:
 *
 * Example configuration node:
 *
 * \verbatim
 *		<AdapterTimeStream name="">
 *			<fixedSizePackets value="true|false"/>
 *			<sampleSize bits=""/>
 *			<samplesPerPacket number=""/>
 *			<packetsPerChunk number=""/>
 *			<samplesPerTimeBlock number=""/>
 *			<subbands number=""/>
 *			<polarisations number=""/>
 *		<\AdapterTimeStream>
 * \verbatim
 *
 * - samplesPerPacket: Number of (time) samples per packet.
 * - fixedSizePackets: Specify if UDP packets are fixed size or not.
 * - sampleSize: Number of bits per sample. (Samples are assumed to be complex
 *               pairs of the number of bits specified).
 * - packetsPerChunk: Number of UDP packets in each input data chunk.
 * - samplesPerTimeBlock: Number of time samples to put in a block.
 * - subbands: Number of sub-bands per packet.
 * - polarisations: Number of polarisations per packet.
 */

class AdapterTimeSeriesDataSet : public AbstractStreamAdapter
{
    private:
        friend class AdapterTimeSeriesDataSetTest;

        typedef float Real;
        typedef std::complex<Real> Complex;

    public:
        /// Constructs a new AdapterTimeStream.
        AdapterTimeSeriesDataSet(const ConfigNode& config);

        /// Destroys the AdapterTimeStream.
        ~AdapterTimeSeriesDataSet() {}

    protected:
        /// Method to deserialise a LOFAR time stream data.
        void deserialise(QIODevice* in);

    private:
        /// Updates and checks the size of the time stream data.
        void _checkData();

        /// Read the udp packet header from a buffer read from the IO device.
        void _readHeader(char* buffer, UDPPacket::Header& header);

        /// Reads the udp data data section into the data blob data array.
        void _readData(unsigned packet, char* buffer,
                TimeSeriesDataSetC32* data);

        /// Prints the header to standard out (for debugging).
        void _printHeader(const UDPPacket::Header& header);

        /// Converts a i8Complex to std::complex float.
        Complex _makeComplex(const TYPES::i8complex& z);

        /// Converts a i8Complex to std::complex float.
        Complex _makeComplex(const TYPES::i16complex& z);

    private:
        /// Constructs an error message with the class name.
        QString _err(const QString& message);

    private:
        TimeSeriesDataSetC32* _timeData;
        bool _fixedPacketSize;
        unsigned _nUDPPacketsPerChunk;
        unsigned _nSamplesPerPacket;
        unsigned _nSamplesPerTimeBlock;
        unsigned _nSubbands;
        unsigned _nPolarisations;
        unsigned _sampleBits;
        unsigned _clock;

        size_t _packetSize;
        size_t _headerSize;
        size_t _packetDataSize;
        size_t _dataSize;
        size_t _paddingSize;
        std::vector<char> _headerTemp;
        std::vector<char> _dataTemp;
        std::vector<char> _paddingTemp;
};


PELICAN_DECLARE_ADAPTER(AdapterTimeSeriesDataSet)

} // namespace lofar
} // namespace pelican
#endif // ADAPTER_TIME_SERIES_DATA_SET
