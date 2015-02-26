#ifndef __ABPACKET_H__
#define __ABPACKET_H__

#include <boost/cstdint.hpp>

namespace pelican {
namespace ampp {

// Size of integration counter
#define INTEGCOUNT_SIZE         6       // Bytes
// Size of payload data
#define PAYLOAD_SIZE            8192    // Bytes
// Size of footer
#define FOOTER_SIZE             8       // Bytes

struct ABPacket {
    struct Header {
        uint8_t integCount[INTEGCOUNT_SIZE];
        uint8_t specQuart;
        uint8_t beam;
    } header;
    char data[PAYLOAD_SIZE];
    char footer[FOOTER_SIZE];
};

}   // namespace ampp
}   // namespace pelican

#endif // __ABPACKET_H__
