#include "BufferingDataClient.h"
#include <QtCore>
#include <QCoreApplication>
#include <boost/bind.hpp>

namespace pelican {
namespace ampp {

template<class DataClientType>
BufferingDataClient<DataClientType>::BufferingDataClient(const ConfigNode& configNode, const DataTypes& types, const Config* config)
    : DataClientType(configNode, types, config)
    , _agent(boost::bind(&DataClientType::getData, this, _1))
    , _halt(false)
{
    // start the thread running that collects data
    _agent.start();

    // dedicate a thread to running the event loop of the agent to ensure messages are delivered to the agent
    QtConcurrent::run(boost::bind(&BufferingDataClient<DataClientType>::exec, this));   
}

template<class DataClientType>
BufferingDataClient<DataClientType>::~BufferingDataClient()
{
    // stop the thread running
    _agent.stop();
    _halt = true;
}

template<class DataClientType>
void BufferingDataClient<DataClientType>::exec() 
{
    while(!_halt) {
        QCoreApplication::processEvents();
    }
}

template<class DataClientType>
pelican::AbstractDataClient::DataBlobHash BufferingDataClient<DataClientType>::getData(pelican::AbstractDataClient::DataBlobHash& hash)
{
    _agent.getData(hash);
    return hash;
}

} // namespace ampp
} // namespace pelican
