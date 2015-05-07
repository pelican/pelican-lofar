#include "ABBufDataClient.h"
#include "ABChunker.h"

using namespace pelican;
using namespace pelican::ampp;

ABBufDataClient::ABBufDataClient(const ConfigNode& configNode,
                const DataTypes& types, const Config* config)
    : PelicanServerClient(configNode, types, config)
{
    // start the thread running
    std::cout << "-------in bdc!=======" << std::endl;
    _agent.start();
}

ABBufDataClient::~ABBufDataClient()
{
    // stop the thread running
    _agent.stop();
}

BufferingAgent::DataBlobHash ABBufDataClient::getData(BufferingAgent::DataBlobHash& hash)
{
    _agent.getData(hash);
    return hash;
}

