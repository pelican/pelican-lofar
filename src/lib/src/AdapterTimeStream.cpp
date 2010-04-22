#include "AdapterTimeStream.h"

#include "LofarUdpHeader.h"

#include "pelican/data/TimeStreamData.h"
#include "pelican/utility/ConfigNode.h"
#include <cmath>

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
    _nTimes = config.getOption("timeSamples", "number", "0").toUInt();
    _dataBytes = config.getOption("dataBytes", "number", "0").toUInt();
}


/**
 * @details
 * Method to deserialise a single station time stream chunk.
 *
 * @param[in] in QIODevice containing a serialised version of a LOFAR
 *               visibility data set.
 */
void AdapterTimeStream::deserialise(QIODevice* in)
{
    // Read the header

    // Read the data
}


/**
 * @details
 */
void AdapterTimeStream::_checkData()
{
    // Check that there is something of to adapt.
    if (_chunkSize == 0) {
        throw QString("AdapterTimeStream: Chunk size Zero.");
    }

    unsigned packetSize = sizeof(UDPPacket);

    // Check the chunk size is a multiple of the udp packet size.
    if (_chunkSize % packetSize != 0) {
        throw QString("AdapterTimeStream: Chunk size '%1' not a multiple of the "
                      " packet size '%2'.").arg(_chunkSize).arg(packetSize);
    }

    // Check the data blob passed to the adapter is allocated.
    if (_data == NULL) {
        throw QString("AdapterTimeStream: Cannot deserialise into an "
                      "unallocated blob!.");
    }

    // If any service data exists update the visibility dimensions from it.
    if (!_serviceData.empty()) _updateDimensions();

    // Check the expected size due to data dimensions matches the chunk size.
    unsigned dataSize = _nTimes * _dataBytes;
    if (dataSize != _chunkSize) {
        throw QString("AdapterTimeStream: Input chunk size does not match "
                "that expected from the adapter configuration. "
                "[TimeStream blob size = %1, chunk size = %2.]")
                .arg(dataSize).arg(_chunkSize);
    }

    // Resize the time stream data blob being read into to match the adapter
    // dimensions.
    _timeData = static_cast<TimeStreamData*>(_data);
    _timeData->resize(_nTimes);
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


} // namespace lofar
} // namespace pelican
