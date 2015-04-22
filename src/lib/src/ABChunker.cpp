#include "pelican/utility/Config.h"
#include <QtNetwork/QUdpSocket>
#include <sys/time.h>
#include <sys/resource.h>
#include <sched.h>
#include <stdio.h>
#include <iostream>

#include "ABChunker.h"

namespace pelican {
namespace ampp {

// Construct the example chunker.
ABChunker::ABChunker(const ConfigNode& config) : AbstractChunker(config)
{
    // Set chunk size from the configuration.
    // The host, port and data type are set in the base class.
    _chunkSize = config.getOption("data", "chunkSize").toInt();
    _pktSize = config.getOption("packet", "size", "8208").toInt();
    _hdrSize = config.getOption("packet", "header", "8").toInt();
    _ftrSize = config.getOption("packet", "footer", "8").toInt();
    _payloadSize = _pktSize - _hdrSize - _ftrSize;
    _pktsPerSpec = config.getOption("spectrum", "packets").toUInt();
    _nPackets = _chunkSize / _pktSize;
    _first = 3;
    _numMissInst = 0;
    _lostPackets = 0;
    _prevSpecQuart = 0;
    _prevIntegCount = 0;
    _chunksProced = 0;
    _savedPktAvailable = 0;
    _savedSpecQuart = 0;
    _savedIntegCount = 0;
    _x = 0;
    _y = 0;

    // Allocate memory for the saved packet
    _pktSaved = new char[_pktSize];

    /* set the CPU affinity of the main thread that reads data off the NIC */
#if 0
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(5, &cpuset);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) > 0)
    {
        std::cerr << "ERROR: Setting affinity failed!" << std::endl;
    }
    if (setpriority(PRIO_PROCESS, 0, -20) < 0)
    {
        std::cerr << "ERROR: Setting priority failed!" << std::endl;
        perror("setpriority");
    }
#endif
}

// Destructor.
ABChunker::~ABChunker()
{
    delete _pktSaved;
}

// Creates a suitable device ready for reading.
QIODevice* ABChunker::newDevice()
{
    // Return an opened QUdpSocket.
    QUdpSocket* socket = new QUdpSocket;
    socket->bind(QHostAddress(host()), port());

    // Wait for the socket to bind.
    while (socket->state() != QUdpSocket::BoundState) {}

    return socket;
}

// Called whenever there is data available on the device.
void ABChunker::next(QIODevice* device)
{
    // Get a pointer to the UDP socket.
    QUdpSocket* socket = static_cast<QUdpSocket*>(device);
    unsigned int specQuart = 0;
    unsigned int beam = 0;
    unsigned long int integCount = 0;
    unsigned int icDiff = 0;
    unsigned int sqDiff = 0;
    unsigned int bytesRead = 0;
    unsigned int lostPackets = 0;
    unsigned int packetCounter = 0;
    unsigned long int missedIntegCount = 0;
    unsigned int missedSpecQuart = 0;
    char pkt[_pktSize];
    char pktMissed[_pktSize];
    char fakeHdr[_hdrSize];

    // Get writable buffer space for the chunk.
    WritableData writableData = getDataStorage(_chunkSize);
    if (writableData.isValid())
    {
        // Get pointer to start of writable memory.
        //char *ptr = (char *) (writableData.ptr());

        // Loop over the number of UDP packets to put in a chunk.
        for (unsigned i = 0; i < _nPackets; i++)
        {
            // Read datagrams for chunk from the UDP socket.
            if (!isActive()) return;

            // Add lost packets that have been unaccounted for in previous
            // chunk. Initially, _lostPackets is 0, so this loop does nothing
            // for the first chunk.
            // Fill in zeros in place of missing packets.
            for (packetCounter = 0; packetCounter < _lostPackets && packetCounter < _nPackets; packetCounter++)
            {
                // Fill in fake header.
                *(fakeHdr + 7) = beam; // Beam number remains the same.
                missedSpecQuart = (_prevSpecQuart + 1) % _pktsPerSpec;
                *(fakeHdr + 6) = missedSpecQuart;
                if (0 == missedSpecQuart)
                {
                    missedIntegCount = _prevIntegCount + 1;
                }
                else
                {
                    missedIntegCount = _prevIntegCount;
                }
                *(fakeHdr + 5) = (unsigned char) (missedIntegCount & 0x00000000000000FF);
                *(fakeHdr + 4) = (unsigned char) ((missedIntegCount & 0x000000000000FF00) >> 8);
                *(fakeHdr + 3) = (unsigned char) ((missedIntegCount & 0x0000000000FF0000) >> 16);
                *(fakeHdr + 2) = (unsigned char) ((missedIntegCount & 0x00000000FF000000) >> 24);
                *(fakeHdr + 1) = (unsigned char) ((missedIntegCount & 0x000000FF00000000) >> 32);
                *(fakeHdr + 0) = (unsigned char) ((missedIntegCount & 0x0000FF0000000000) >> 40);
                (void) memcpy(pktMissed, fakeHdr, _hdrSize);
                // Fill in zeros in place of data and footer.
                (void) memset(pktMissed + _hdrSize, '\0', _payloadSize);// + _ftrSize);
                //std::cout << missedIntegCount << std::endl;
                writableData.write(pktMissed, _pktSize, bytesRead);
                bytesRead += _pktSize;
                _prevSpecQuart = missedSpecQuart;
                _prevIntegCount = missedIntegCount;
            }
            _prevPktCount += packetCounter;
            // packetCounter is now either _lostPackets (which could be 0) or
            // _nPackets.
            i += packetCounter;
            if (packetCounter != 0)
            {
                std::cerr << packetCounter << " packets added at beginning of chunk." << std::endl;
            }
            _lostPackets -= packetCounter;
            // Check if the chunk is now full.
            if (packetCounter == _nPackets)
            {
                break;
            }
            // If chunk is not full, read data off the socket. At this point,
            // _lostPackets has to be 0, i.e., we have dealt with residual lost
            // packets from the previous chunk and you still have space in this
            // chunk.
            Q_ASSERT(0 == _lostPackets);

            // Now write the packet saved from the last chunk
            if (_savedPktAvailable)
            {
                //std::cout << _savedIntegCount << std::endl;
                writableData.write(_pktSaved, _pktSize, bytesRead);
                bytesRead += _pktSize;

                Q_ASSERT(_savedSpecQuart == (_prevSpecQuart + 1) % _pktsPerSpec);
                Q_ASSERT(0 == _savedSpecQuart ?
                         _savedIntegCount == _prevIntegCount + 1 :
                         _savedIntegCount == _prevIntegCount);
                // Update previous counts.
                _prevSpecQuart = _savedSpecQuart;
                _prevIntegCount = _savedIntegCount;
                _prevPktCount++;

                _savedPktAvailable = 0;
                i++;
                // Check if the chunk is now full.
                if (i == _nPackets)
                {
                    break;
                }
            }

            // Read the datagram, but avoid using pendingDatagramSize().
            while (!socket->hasPendingDatagrams()) {
                // MUST WAIT for the next datagram.
                socket->waitForReadyRead(100);
            }

            // Read the current packet from the socket
            unsigned int len = socket->readDatagram(pkt, _pktSize);
            if (len != _pktSize)
            {
                std::cerr << "ERROR: readDatagram() <= 0!" << std::endl;
                continue;
            }

            // Get the packet integration count
            unsigned char *buf = (unsigned char *) pkt;
            unsigned long int counter = (*((unsigned long int *) buf))
                                        & 0x0000FFFFFFFFFFFF;
            integCount = (unsigned long int)        // Casting required.
                          (((counter & 0x0000FF0000000000) >> 40)
                         + ((counter & 0x000000FF00000000) >> 24)
                         + ((counter & 0x00000000FF000000) >> 8)
                         + ((counter & 0x0000000000FF0000) << 8)
                         + ((counter & 0x000000000000FF00) << 24)
                         + ((counter & 0x00000000000000FF) << 40));

            // Get the spectral quarter number
            //specQuart = (unsigned char) currPacket.header.specQuart;
            specQuart = (unsigned int) buf[6];

            // Get the beam number
            beam = (unsigned int) buf[7];

            if (_first > 1)
            {
                // Ignore the first <= 7 packets
                if (0 == specQuart)
                {
                    _first--;
                }
                if (_first > 1)
                {
                    _prevSpecQuart = specQuart;
                    _prevIntegCount = integCount;
                    //_prevPktCount = (specQuart * _pktsPerSpec) + integCount;
                    std::cout << "Spectral quarter = " << specQuart << ". Skipping..." << std::endl;
                    continue;
                }
            }

#if 0
            // Check for missed packets
            if (((_prevSpecQuart + 1) % 4) != specQuart)
            {
                icDiff = integCount - _prevIntegCount;
                if (0 == icDiff)    // same integration, different spectral quarter
                {
                    sqDiff = specQuart - _prevSpecQuart;
                    _numMissInst++;
                    lostPackets = (sqDiff - 1);
                    std::cerr << lostPackets << " packets dropped!" << std::endl;
                }
                else                // different integration
                {
                    _numMissInst++;
                    lostPackets = ((_pktsPerSpec - 1 - _prevSpecQuart)
                                    + _pktsPerSpec * (icDiff - 1)
                                    + specQuart);
                    std::cerr << lostPackets << " packets dropped!" << std::endl;
                }
            }
            if (0 == specQuart)
            {
                icDiff = integCount - _prevIntegCount;
                if (icDiff != 1)
                {
                    _numMissInst++;
                    lostPackets = ((_pktsPerSpec - 1 - _prevSpecQuart)
                                    + _pktsPerSpec * (icDiff - 1)
                                    + specQuart);
                    std::cerr << lostPackets << " packets dropped!" << std::endl;
                }
            }
#endif

            unsigned long int pktCount = (integCount * _pktsPerSpec) + specQuart;
            if (_first != 1)
            {
                lostPackets = pktCount - _prevPktCount - 1;
                if (lostPackets != 0)
                {
                    std::cerr << lostPackets << " packets dropped!" << std::endl;
                }
            }
            else
            {
                _prevPktCount = pktCount - 1;
                _first = 0;
            }

            // Fill in zeros in place of missing packets.
            for (packetCounter = 0; packetCounter < lostPackets && i + packetCounter < _nPackets; packetCounter++)
            {
                // Fill in fake header.
                *(fakeHdr + 7) = beam; // Beam number remains the same.
                missedSpecQuart = (_prevSpecQuart + 1) % _pktsPerSpec;
                *(fakeHdr + 6) = missedSpecQuart;
                if (0 == missedSpecQuart)
                {
                    missedIntegCount = _prevIntegCount + 1;
                }
                else
                {
                    missedIntegCount = _prevIntegCount;
                }
                *(fakeHdr + 5) = (unsigned char) (missedIntegCount & 0x00000000000000FF);
                *(fakeHdr + 4) = (unsigned char) ((missedIntegCount & 0x000000000000FF00) >> 8);
                *(fakeHdr + 3) = (unsigned char) ((missedIntegCount & 0x0000000000FF0000) >> 16);
                *(fakeHdr + 2) = (unsigned char) ((missedIntegCount & 0x00000000FF000000) >> 24);
                *(fakeHdr + 1) = (unsigned char) ((missedIntegCount & 0x000000FF00000000) >> 32);
                *(fakeHdr + 0) = (unsigned char) ((missedIntegCount & 0x0000FF0000000000) >> 40);
                (void) memcpy(pktMissed, fakeHdr, _hdrSize);
                // Fill in zeros in place of data and footer.
                (void) memset(pktMissed + _hdrSize, '\0', _payloadSize);// + _ftrSize);
                //std::cout << missedIntegCount << std::endl;
                writableData.write(pktMissed, _pktSize, bytesRead);
                bytesRead += _pktSize;
                _prevSpecQuart = missedSpecQuart;
                _prevIntegCount = missedIntegCount;
            }
            _prevPktCount += packetCounter;
            if (packetCounter != 0)
            {
                std::cerr << packetCounter << " packets added." << std::endl;
            }
            i += packetCounter;

            // This is either 0 or the end of the chunk was reached first so
            // there are empty packets to fill in the next chunk.
            _lostPackets = lostPackets - packetCounter;

            // Write out current packet.
            if (i < _nPackets)
            {
                Q_ASSERT(bytesRead <= (_chunkSize - _pktSize));
                //std::cout << integCount << std::endl;
                writableData.write(pkt, _pktSize, bytesRead);
                bytesRead += _pktSize;

                // Update previous counts.
                _prevSpecQuart = specQuart;
                _prevIntegCount = integCount;
                _prevPktCount++;
            }
            else    // i == _nPackets
            {
                // Save the current packet so that it will be written in the
                // next available chunk.
                (void) memcpy(_pktSaved, pkt, _pktSize);
                _savedPktAvailable = 1;
                _savedSpecQuart = specQuart;
                _savedIntegCount = integCount;

                // Update previous counts.
                //_prevSpecQuart = missedSpecQuart;
                //_prevIntegCount = missedIntegCount;
            }
        }
        _chunksProced++;
        _y++;
        if (_y % 100 == 0)
        {
            std::cout << _chunksProced << " chunks processed." << std::endl;
        }
    }
    // Must discard the datagram if there is no available space.
    else
    {
        _x++;
        if (_x % 100 == 0)
        {
            std::cout << "100x no available space!" << std::endl;
        }
        socket->readDatagram(0, 0);
    }
}

} // namespace ampp
} // namespace pelican

