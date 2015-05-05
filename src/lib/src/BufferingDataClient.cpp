#include "BufferingDataClient.h"


namespace pelican {
namespace ampp {


template<class DataClientType>
BufferingDataClient<DataClientType>::BufferingDataClient(DataClientType& client)
    , _agent(client)
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
DataBlobHash BufferingDataClient<DataClientType>::getData(DataBlobHash& hash)
{
    _agent.getData(hash);
    return hash;
}

} // namespace ampp
} // namespace pelican
