#ifndef ABBUFDATACLIENT_H
#define ABBUFDATACLIENT_H

#include "pelican/core/PelicanServerClient.h"
#include "BufferingDataClient.h"

namespace pelican {
namespace ampp {

typedef BufferingDataClient<PelicanServerClient> ABBufDataClient;

PELICAN_DECLARE_CLIENT(ABBufDataClient)

} // namespace ampp
} // namespace pelican

#endif // ABBUFDATACLIENT_H

