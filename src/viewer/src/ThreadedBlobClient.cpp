#include "viewer/ThreadedBlobClient.h"
#include "lib/PelicanBlobClient.h"
#include "lib/SpectrumDataSet.h"

#include <QtCore/QCoreApplication>

#include <iostream>

namespace pelican {

namespace ampp {


/**
 *@details ThreadedBlobClient
 */
ThreadedBlobClient::ThreadedBlobClient(const QString& host, quint16 port,
        const QString& stream, QObject* parent)
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
    _client = new PelicanBlobClient(_dataStream, _host, _port);
    SpectrumDataSetStokes blob;
    SpectrumDataSetStokes lastBlob;
    QHash<QString, DataBlob*> dataHash;
    dataHash.insert(_dataStream, &blob);
    while( _isRunning )
    {
        try {
            _client->getData(dataHash);
            lastBlob = blob;
            emit dataUpdated(_dataStream, &lastBlob);
            QCoreApplication::processEvents();
        }
        catch (QString e) {
            std::cout << "ThreadedBlobClient::run(): ERROR: "
                    << e.toStdString() << std::endl;
        }
    }
    delete _client;
}

} // namespace ampp
} // namespace pelican
