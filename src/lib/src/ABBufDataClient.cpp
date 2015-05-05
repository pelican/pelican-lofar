#include "ABBufDataClient.h"
#include "ABChunker.h"

using namespace pelican;
using namespace pelican::ampp;

ABBufDataClient::ABBufDataClient(const ConfigNode& configNode,
                const DataTypes& types, const Config* config)
    : PelicanServerClient(configNode, types, config)
{
}

ABBufDataClient::~ABBufDataClient()
{
}
