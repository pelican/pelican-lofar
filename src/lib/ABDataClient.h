#ifndef ABDATACLIENT_H
#define ABDATACLIENT_H

//#include "pelican/core/DirectStreamDataClient.h"
#include "pelican/core/PelicanServerClient.h"

namespace pelican {
namespace ampp {

//class ABDataClient : public DirectStreamDataClient
class ABDataClient : public PelicanServerClient
{
    public:
        ABDataClient( const ConfigNode& configNode,
                 const DataTypes& types, const Config* config );
        ~ABDataClient();
    private:
};

PELICAN_DECLARE_CLIENT(ABDataClient)

} // namespace ampp
} // namespace pelican

#endif // ABDATACLIENT_H

