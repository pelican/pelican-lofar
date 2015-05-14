#include "K7Packet.h"


namespace pelican {

namespace ampp {


K7Packet::K7Packet()
{
}

K7Packet::~K7Packet()
{
}

QIODevice& operator>>(K7PacketHeader& packet_header, QIODevice& device)
{
    
    while (device.bytesAvailable() < sizeof(K7PacketHeader))
    {
        device->waitForReadyRead(-1);
    }

    device->read(&packet_header, sizeof(K7PacketHeader));

    return s;
}

} // namespace ampp
} // namespace pelican
