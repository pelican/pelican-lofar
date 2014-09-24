#include "JTestChunker.h"
#include "pelican/utility/Config.h"
#include <QtNetwork/QUdpSocket>

namespace pelican {
namespace ampp {

// Construct the example chunker.
JTestChunker::JTestChunker(const ConfigNode& config) : AbstractChunker(config)
{
    // Set chunk size from the configuration.
    // The host, port and data type are set in the base class.
    _chunkSize = config.getOption("data", "chunkSize").toInt();
}

// Creates a suitable device ready for reading.
QIODevice* JTestChunker::newDevice()
{
    // Return an opened QUdpSocket.
    QUdpSocket* socket = new QUdpSocket;
    socket->bind(QHostAddress(host()), port());

    // Wait for the socket to bind.
    while (socket->state() != QUdpSocket::BoundState) {}
    return socket;
}

// Called whenever there is data available on the device.
void JTestChunker::next(QIODevice* device)
{
    // Get a pointer to the UDP socket.
    QUdpSocket* udpSocket = static_cast<QUdpSocket*>(device);
    _bytesRead = 0;

    // Get writable buffer space for the chunk.
    WritableData writableData = getDataStorage(_chunkSize);
    if (writableData.isValid()) {
        // Get pointer to start of writable memory.
        char* ptr = (char*) (writableData.ptr());

        // Read datagrams for chunk from the UDP socket.
        while (isActive() && _bytesRead < _chunkSize) {
            // Read the datagram, but avoid using pendingDatagramSize().
            if (!udpSocket->hasPendingDatagrams()) {
                // MUST WAIT for the next datagram.
                udpSocket->waitForReadyRead(100);
                continue;
            }
            qint64 maxlen = _chunkSize - _bytesRead;
            qint64 length = udpSocket->readDatagram(ptr + _bytesRead, maxlen);
            if (length > 0)
                _bytesRead += length;
        }
    }

    // Must discard the datagram if there is no available space.
    else {
        udpSocket->readDatagram(0, 0);
    }
}

} // namespace ampp
} // namespace pelican

