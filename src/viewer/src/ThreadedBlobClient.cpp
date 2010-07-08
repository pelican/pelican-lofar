#include <QCoreApplication>
#include <iostream>
#include "viewer/ThreadedBlobClient.h"
#include "lib/PelicanBlobClient.h"
#include "lib/ChannelisedStreamData.h"


namespace pelican {

namespace lofar {


/**
 *@details ThreadedBlobClient 
 */
ThreadedBlobClient::ThreadedBlobClient( const QString& host, quint16 port, const QString& stream, QObject* parent )
    : QThread(parent), _host(host), _port(port), _dataStream(stream)
{
    _isRunning = false;
    start();
}

/**
 *@details
 */
ThreadedBlobClient::~ThreadedBlobClient()
{
    _isRunning = false;
    wait();
}

void ThreadedBlobClient::run()
{
    _isRunning = true;
    _client = new PelicanBlobClient( _dataStream, _host, _port );
    ChannelisedStreamData blob;
    ChannelisedStreamData lastBlob;
    QHash<QString, DataBlob*> dataHash;
    dataHash.insert(_dataStream, &blob);
    while( _isRunning )
    {
        _client->getData(dataHash);
        lastBlob = blob;
        emit dataUpdated(_dataStream, &lastBlob);
        QCoreApplication::processEvents();
    }
    delete _client;
}

} // namespace lofar
} // namespace pelican
