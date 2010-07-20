// Pelican stuff
#include "pelican/comms/PelicanClientProtocol.h"
#include "pelican/comms/StreamDataRequest.h"
#include "pelican/data/DataRequirements.h"
#include "pelican/comms/DataBlobResponse.h"
#include "pelican/comms/ServerResponse.h"

// Pelican-Lofar stuff
#include "PelicanBlobClient.h"
#include "ChannelisedStreamData.h"
//#include "SubbandSpectra.h" // FIXME for new data blob.

// QT stuff
#include <QtCore/QByteArray>

// Boost stuff
#include <boost/shared_ptr.hpp>

// C++ stuff
#include <iostream>

namespace pelican {
namespace lofar {

// class PelicanBlobClient
PelicanBlobClient::PelicanBlobClient(QString blobType, QString server, quint16 port)
    : _server(server), _blobType(blobType), _port(port)
{
    _protocol = new PelicanClientProtocol;
    _tcpSocket = new QTcpSocket;
}

// Destructor
PelicanBlobClient::~PelicanBlobClient()
{
    delete _tcpSocket;
    delete _protocol;
}

// Read data from the Pelican Lofar pipeline
void PelicanBlobClient::getData(QHash<QString, DataBlob*>& dataHash)
{
//    std::cout << "PelicanBlobClient::getData()" << std::endl;

    // Check that we are still connected to server, if not reconnect
    if (_tcpSocket->state() == QAbstractSocket::UnconnectedState) {
        std::cout << "PelicanBlobClient: Disconnected from server, reconnecting."
                  << std::endl;
        connectToLofarPelican();
    }

    // Get a pointer to the data blob from the hash.
    // FIXME for new stokes data blob.
    SubbandSpectraStokes* blob = (SubbandSpectraStokes*) dataHash["SubbandSpectraStokes"];
    //ChannelisedStreamData* blob = (ChannelisedStreamData*) dataHash["ChannelisedStreamData"];

    // Wait for data to be available to socket, and read
    _tcpSocket->waitForReadyRead();
    boost::shared_ptr<ServerResponse> r = _protocol->receive(*_tcpSocket);
//    std::cout << "PelicanBlobClient::getData(): type = " << r->type() << std::endl;

    // Check what type of response we have
    switch(r->type()) {
        case ServerResponse::Error:  // Error occurred!!
            std::cerr << "PelicanBlobClient: Server Error: '"
                      << r->message().toStdString() << "'" << std::endl;
            break;
        case ServerResponse::Blob:   // We have received a blob
            {
                DataBlobResponse* res = static_cast<DataBlobResponse*>(r.get());
                while (_tcpSocket->bytesAvailable() < (qint64)res->dataSize())
                   _tcpSocket -> waitForReadyRead(-1);
                blob->deserialise(*_tcpSocket, res->byteOrder());
            }
            break;
        case ServerResponse::StreamData:   // We have stream data
            std::cout << "Stream Data" << std::endl;
            break;
        case ServerResponse::ServiceData:  // We have service data
            std::cout << "Service Data" << std::endl;
            break;
        default:
            std::cerr << "PelicanBlobClient: Unknown Response" << std::endl;
            break;
    }
}

// Connect to Pelican Lofar and register requested data type
void PelicanBlobClient::connectToLofarPelican()
{
    while (_tcpSocket->state() == QAbstractSocket::UnconnectedState) {

        _tcpSocket->connectToHost(_server, _port);

        if (!_tcpSocket -> waitForConnected(5000) ||
                _tcpSocket -> state() == QAbstractSocket::UnconnectedState)
        {
            std::cerr << "PelicanBlobClient: Unable to connect to server ("
                      << _server.toStdString() << ":" << _port << ")"
                      << std::endl;
            sleep(2);
            continue;
        }

        // Define the data type which the client will except and send request
        StreamDataRequest req;
        DataRequirements require;
        require.setStreamData(_blobType);
        req.addDataOption(require);

        QByteArray data = _protocol->serialise(req);
        _tcpSocket -> write(data);
        _tcpSocket -> waitForBytesWritten(data.size());
        _tcpSocket -> flush();
    }
}

} //namespace lofar
} // namespace pelican
