#ifndef ABBUFDATACLIENT_H
#define ABBUFDATACLIENT_H

#include "pelican/core/PelicanServerClient.h"
#include "BufferingAgent.h"
#include <deque>

namespace pelican {
namespace ampp {

class ABBufDataClient : public PelicanServerClient
{
    public:
        ABBufDataClient( const ConfigNode& configNode,
                 const DataTypes& types, const Config* config );
        ~ABBufDataClient();

        // override the getData method only
        BufferingAgent::DataBlobHash getData(BufferingAgent::DataBlobHash&);

    private:
        BufferingAgent _agent;
};

PELICAN_DECLARE_CLIENT(ABBufDataClient)

} // namespace ampp
} // namespace pelican

#endif // ABBUFDATACLIENT_H

