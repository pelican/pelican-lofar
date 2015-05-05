#ifndef ABBUFDATACLIENT_H
#define ABBUFDATACLIENT_H

#include "pelican/core/PelicanServerClient.h"

namespace pelican {
namespace ampp {

class ABBufDataClient : public PelicanServerClient
{
    public:
        ABBufDataClient( const ConfigNode& configNode,
                 const DataTypes& types, const Config* config );
        ~ABBufDataClient();
    private:
};

PELICAN_DECLARE_CLIENT(ABBufDataClient)

} // namespace ampp
} // namespace pelican

#endif // ABBUFDATACLIENT_H

