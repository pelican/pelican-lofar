#ifndef ADAPTERTIMESTREAM_H
#define ADAPTERTIMESTREAM_H

#include "pelican/adapters/AbstractStreamAdapter.h"

/**
 * @file AdapterTimeStream.h
 */

namespace pelican {

class ConfigNode;
class TimeStreamData;

namespace lofar {


/**
 * @class AdapterTimeStream
 *
 * @brief
 * Adapter to deserialise time stream data chunks from a lofar station.
 *
 * @details
 * Stream adapter to deserialise time stream data chunks from a lofar station.
 */

class AdapterTimeStream : public AbstractStreamAdapter
{
    private:
        friend class AdapterTimeStreamTest;

    public:
        /// Constructs a new AdapterTimeStream.
        AdapterTimeStream(const ConfigNode& config);

        /// Destroys the AdapterTimeStream.
        ~AdapterTimeStream() {}

    protected:
        /// Method to deserialise a LOFAR time stream data.
        void deserialise(QIODevice* in);

    private:
        /// Updates and checks the size of the time stream data.
        void _checkData();

        /// Updates dimensions of the time stream data being deserialised.
        void _updateDimensions();

    private:
        TimeStreamData* _timeData;
        unsigned _nTimes;
        unsigned _dataBytes;


};

} // namespace lofar
} // namespace pelican

#endif // ADAPTERTIMESTREAM_H
