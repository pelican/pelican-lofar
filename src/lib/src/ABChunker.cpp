#include "pelican/utility/Config.h"
#include <QtNetwork/QUdpSocket>
#include <sys/time.h>
#include <sys/resource.h>
#include <sched.h>
#include <stdio.h>

#include "ABChunker.h"
#include "ABPacket.h"

namespace pelican {
namespace ampp {

// Construct the example chunker.
ABChunker::ABChunker(const ConfigNode& config) : AbstractChunker(config)
{
    // Set chunk size from the configuration.
    // The host, port and data type are set in the base class.
    _chunkSize = config.getOption("data", "chunkSize").toInt();
    _pktSize = config.getOption("packet", "size").toInt();
    _hdrSize = config.getOption("packet", "header").toInt();
    _ftrSize = config.getOption("packet", "footer").toInt();
    _payloadSize = _pktSize - _hdrSize - _ftrSize;
    _pktsPerSpec = config.getOption("spectrum", "packets").toUInt();
    _nPackets = _chunkSize / _pktSize;
    _first = 2;
    _numMissInst = 0;
    _numMissPkts = 0;
    _chunksProced = 0;
    _x = 0;
    _y = 0;

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
    ABPacket currPacket;
    signed int specQuart = 0;
    static signed int prevSpecQuart = 0;
    unsigned long int integCount = 0;
    static unsigned long int prevIntegCount = 0;
    unsigned int icDiff = 0;
    unsigned int sqDiff = 0;
    unsigned int bytesRead = 0;
    char pkt[_pktSize];
    char fakeHdr[_hdrSize];

    // Get writable buffer space for the chunk.
    WritableData writableData = getDataStorage(_chunkSize);
    if (writableData.isValid())
    {
        // Get pointer to start of writable memory.
        char *ptr = (char *) (writableData.ptr());

        // Loop over the number of UDP packets to put in a chunk
        for (unsigned i = 0; i < _nPackets; i++)
        {
            // Read datagrams for chunk from the UDP socket.
            if (!isActive()) return;

            // Read the datagram, but avoid using pendingDatagramSize().
            while (!socket->hasPendingDatagrams()) {
                // MUST WAIT for the next datagram.
                socket->waitForReadyRead(100);
            }

            // Read the current packet from the socket
#if 0
            ABPacket *curPkt = (ABPacket*) (ptr + bytesRead);
            unsigned int len = socket->readDatagram(reinterpret_cast<char*>(&currPacket), _pktSize);
#endif
            //unsigned int len = socket->readDatagram(ptr + bytesRead, _pktSize);
            unsigned int len = socket->readDatagram(pkt, _pktSize);
            if (len != _pktSize)
            {
                std::cerr << "ERROR: readDatagram() <= 0!" << std::endl;
                continue;
            }

            // Get the packet integration count
            //unsigned long int counter = (*((unsigned long int *) currPacket.header.integCount))
            //unsigned char *count = (unsigned char *) (ptr + bytesRead - len);
            unsigned char *count = (unsigned char *) pkt;
            unsigned long int counter = (*((unsigned long int *) count))
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
            specQuart = (unsigned char) count[6];
            if (_first)
            {
                // Ignore the first <= 7 packets
                if (0 == specQuart)
                {
                    _first--;
                }
                if (_first)
                {
                    prevSpecQuart = specQuart;
                    prevIntegCount = integCount;
                    std::cout << "Spectral quarter = " << specQuart << ". Skipping..." << std::endl;
                    continue;
                }
            }

            // Check for missed packets
            if (((prevSpecQuart + 1) % 4) != specQuart)
            {
                icDiff = integCount - prevIntegCount;
                if (0 == icDiff)    // same integration, different spectral quarter
                {
                    sqDiff = specQuart - prevSpecQuart;
                    _numMissInst++;
                    _numMissPkts = (sqDiff - 1);
                    std::cerr << _numMissPkts << " packets dropped!" << std::endl;
                }
                else                // different integration
                {
                    _numMissInst++;
                    _numMissPkts = ((_pktsPerSpec - 1 - prevSpecQuart)
                                    + _pktsPerSpec * (icDiff - 1)
                                    + specQuart);
                    std::cerr << _numMissPkts << " packets dropped!" << std::endl;
                }
            }
            if (0 == specQuart)
            {
                icDiff = integCount - prevIntegCount;
                if (icDiff != 1)
                {
                    _numMissInst++;
                    _numMissPkts = ((_pktsPerSpec - 1 - prevSpecQuart)
                                    + _pktsPerSpec * (icDiff - 1)
                                    + specQuart);
                    std::cerr << _numMissPkts << " packets dropped!" << std::endl;
                }
            }
            prevSpecQuart = specQuart;
            prevIntegCount = integCount;

#if 0
            // Fill in zeros in place of missing packets.
            for (unsigned j = 0; j < _numMissPkts; j++)
            {
                // Fill in fake header.
                /*(ptr + 5) = (unsigned char) (_counter & 0x00000000000000FF);
                *(ptr + 4) = (unsigned char) ((_counter & 0x000000000000FF00) >> 8);
                *(ptr + 3) = (unsigned char) ((_counter & 0x0000000000FF0000) >> 16);
                *(ptr + 2) = (unsigned char) ((_counter & 0x00000000FF000000) >> 24);
                *(ptr + 1) = (unsigned char) ((_counter & 0x000000FF00000000) >> 32);
                *(ptr + 0) = (unsigned char) ((_counter & 0x0000FF0000000000) >> 40);
                *(ptr + 6) = _specQuart; // Spectral quarter.
                *(ptr + 7) = _beam; // Beam number.*/
                (void) memcpy(ptr + bytesRead, fakeHdr, _hdrSize);
                bytesRead += _hdrSize;
                // Fill in zeros in place of data.
                (void) memcpy(ptr + bytesRead, '\0', _payloadSize);
                bytesRead += _payloadSize;
                // Fill in fake footer.
                (void) memcpy(ptr + bytesRead, '\0', _ftrSize);
                bytesRead += _ftrSize;
            }
            i += _numMissPkts;
#endif
            // Write out current packet.
            (void) memcpy(ptr + bytesRead, pkt, _pktSize);
            bytesRead += _pktSize;
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

