#ifndef JTESTDATACLIENT_H
#define JTESTDATACLIENT_H

#include "pelican/core/DirectStreamDataClient.h"

namespace pelican {
namespace ampp {

class JTestDataClient : public DirectStreamDataClient
{
    public:
        JTestDataClient( const ConfigNode& configNode,
                 const DataTypes& types, const Config* config );
        ~JTestDataClient();
    private:
};

PELICAN_DECLARE_CLIENT(JTestDataClient)

} // namespace ampp
} // namespace pelican

#endif // JTESTDATACLIENT_H

