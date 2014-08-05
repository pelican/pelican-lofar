#ifndef EMBRACE_SUBBAND_SPLITTING_CHUNKER_H_
#define EMBRACE_SUBBAND_SPLITTING_CHUNKER_H_

/**
 * @file EmbraceSubbandSplittingChunker.h
 */

#include "LofarTypes.h"
#include "LofarUdpHeader.h"

#include "pelican/server/AbstractChunker.h"

#include <QtCore/QString>
#include <QtCore/QObject>
#include <QtCore/QMutex>

namespace pelican {
namespace ampp {

class DataManager;

/**
 * @class EmbraceSubbandSplittingChunker
 *
 * @ingroup pelican_lofar
 *
 * @brief
 *
 * @details
 */

class EmbraceSubbandSplittingChunker : public AbstractChunker
{
    public:
        /// Constructor
        EmbraceSubbandSplittingChunker(const ConfigNode& config);

        /// Destructor
        ~EmbraceSubbandSplittingChunker() {}

        /// Creates the socket to use for the incoming data stream.
        virtual QIODevice* newDevice();

        /// Called whenever there is data ready to be processed.
        virtual void next(QIODevice*);

        /// Sets the number of packets to read.
        void setPackets(int packets) { _nPackets = packets; }

    private:
        /// Generates an empty UDP packet.
        void updateEmptyPacket(UDPPacket& packet, unsigned seqid,
                unsigned blockid);

        /// Write UDPPacket to writeableData object
        int writePacket(WritableData* writer, UDPPacket& packet,
                unsigned packetSize, unsigned offset);

        /// Returns an error message suitable for throwing.
        QString _err(QString message)
        { return QString("EmbraceSubbandSplittingChunker::") + message; }

    private:
        //QMutex _mutex;
        unsigned _nPackets;
        unsigned _packetsRejected;
        unsigned _packetsAccepted;

        // Packet dimensions.
        unsigned _nSamples;
        unsigned _nSubbands;
        unsigned _stream1Subbands;
        unsigned _stream2Subbands;
        unsigned _stream1SubbandStart;
        unsigned _stream1SubbandEnd;
        unsigned _stream2SubbandStart;
        unsigned _stream2SubbandEnd;
        unsigned _byte1OfStream1;
        unsigned _byte1OfStream2;

        unsigned _nPolarisations;

        unsigned _packetSize;
        unsigned _packetSizeStream1;
        unsigned _packetSizeStream2;
        unsigned _bytesStream1;
        unsigned _bytesStream2;
        unsigned _startTime;
        unsigned _startBlockid;
        unsigned _clock;

        UDPPacket _emptyPacket;
        UDPPacket _emptyPacket1;
        UDPPacket _emptyPacket2;

        friend class EmbraceSubbandSplittingChunkerTest;
};

PELICAN_DECLARE_CHUNKER(EmbraceSubbandSplittingChunker)

} // namespace ampp
} // namespace pelican
#endif // EMBRACE_SUBBAND_SPLITTING_CHUNKER_H_
