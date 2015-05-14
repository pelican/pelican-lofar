#ifndef K7PACKET_H
#define K7PACKET_H

#include <time.h>
/**
 * @file K7Packet.h
 */

namespace pelican {

namespace ampp {

/**
 * @class K7Packet
 *  
 * @brief
 * 
 * @details
 * 
 */

struct K7PacketHeader {
    time_t  _utc_timestamp;
    uint_32 _integration_counter;
    uint_32 _number_of_integrations; // per spectrum

    friend QIOStream& operator>>(K7PacketHeader&, QIOStream&);
};

class K7Packet
{
    public:
        K7Packet();
        ~K7Packet();

    private:
        K7PacketHeader _header;

    friend QIOStream& operator>>(K7Packet &, QIOStream&);
};


} // namespace ampp
} // namespace pelican
#endif // K7PACKET_H 
