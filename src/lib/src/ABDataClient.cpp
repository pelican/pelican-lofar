#include "ABDataClient.h"
#include "ABChunker.h"

using namespace pelican;
using namespace pelican::ampp;

#if 0
ABDataClient::ABDataClient(const ConfigNode& configNode,
                const DataTypes& types, const Config* config)
    : DirectStreamDataClient(configNode, types, config)
{
    addStreamChunker("ABChunker");
}
#endif
ABDataClient::ABDataClient(const ConfigNode& configNode,
                const DataTypes& types, const Config* config)
    : PelicanServerClient(configNode, types, config)
{
}

ABDataClient::~ABDataClient()
{
}
