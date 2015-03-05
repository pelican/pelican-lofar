#include "ABDataClient.h"
#include "ABChunker.h"

using namespace pelican;
using namespace pelican::ampp;

ABDataClient::ABDataClient(const ConfigNode& configNode,
                const DataTypes& types, const Config* config)
    : PelicanServerClient(configNode, types, config)
{
}

ABDataClient::~ABDataClient()
{
}
