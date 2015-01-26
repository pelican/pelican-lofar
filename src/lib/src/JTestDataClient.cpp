#include "JTestDataClient.h"
#include "JTestChunker.h"

using namespace pelican;
using namespace pelican::ampp;

JTestDataClient::JTestDataClient(const ConfigNode& configNode,
                const DataTypes& types, const Config* config)
    : DirectStreamDataClient(configNode, types, config)
{
    addStreamChunker("JTestChunker");
}

JTestDataClient::~JTestDataClient()
{
}

