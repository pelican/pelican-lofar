#include "BufferingDataClient.h"
#include <boost/bind.hpp>


namespace pelican {
namespace ampp {

template<class DataClientType>
BufferingDataClient<DataClientType>::BufferingDataClient(const ConfigNode& configNode, const DataTypes& types, const Config* config)
    : DataClientType(configNode, types, config)
    , _agent(boost::bind(&DataClientType::getData, this, _1))
{
    // start the thread running
    _agent.start();
}

template<class DataClientType>
BufferingDataClient<DataClientType>::~BufferingDataClient()
{
    // stop the thread running
    _agent.stop();
}

template<class DataClientType>
pelican::AbstractDataClient::DataBlobHash BufferingDataClient<DataClientType>::getData(pelican::AbstractDataClient::DataBlobHash& hash)
{
    _agent.getData(hash);
    return hash;
}

} // namespace ampp
} // namespace pelican
