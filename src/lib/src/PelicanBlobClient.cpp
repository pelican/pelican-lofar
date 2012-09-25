// Pelican headers.
#include "pelican/comms/PelicanClientProtocol.h"
#include "pelican/comms/StreamDataRequest.h"
#include "pelican/data/DataSpec.h"
#include "pelican/comms/DataBlobResponse.h"
#include "pelican/comms/ServerResponse.h"

// Pelican-Lofar headers.
#include "PelicanBlobClient.h"
#include "SpectrumDataSet.h"

// QT headers.
#include <QtCore/QByteArray>

// Boost headers.
#include <boost/shared_ptr.hpp>

// C++ headers.
#include <iostream>

using std::cout;
using std::endl;
using std::cerr;

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
    // Check that we are still connected to server, if not reconnect
    if (_tcpSocket->state() == QAbstractSocket::UnconnectedState) {
        cout << "PelicanBlobClient: Disconnected from server, reconnecting." << endl;
        connectToLofarPelican();
    }

    // Get a pointer to the data blob from the hash.
    SpectrumDataSetStokes* blob = (SpectrumDataSetStokes*) dataHash["SpectrumDataSetStokes"];

    // Wait for data to be available to socket, and read
    _tcpSocket->waitForReadyRead();
    boost::shared_ptr<ServerResponse> response = _protocol->receive(*_tcpSocket);

    // Check what type of response we have
    switch(response->type())
    {
        case ServerResponse::Error:  // Error occurred!!
        {
            cerr << "PelicanBlobClient: Server Error: '"
                 << response->message().toStdString() << "'" << endl;
            break;
        }
        case ServerResponse::Blob: // We have received a blob
        {
            DataBlobResponse* res = (DataBlobResponse*)response.get();
            while (_tcpSocket->bytesAvailable() < (qint64)res->dataSize())
                _tcpSocket -> waitForReadyRead(-1);
            blob->deserialise(*_tcpSocket, res->byteOrder());
            break;
        }
        case ServerResponse::StreamData: // We have stream data
        {
            cout << "Stream Data" << endl;
            break;
        }
        case ServerResponse::ServiceData:  // We have service data
        {
            cout << "Service Data" << endl;
            break;
        }
        default:
            cerr << "PelicanBlobClient: Unknown Response" << endl;
            break;
    }
}


// Connect to Pelican Lofar and register requested data type
void PelicanBlobClient::connectToLofarPelican()
{
    while (_tcpSocket->state() == QAbstractSocket::UnconnectedState) {

        _tcpSocket->connectToHost(_server, _port);

        if (!_tcpSocket->waitForConnected(5000) ||
                _tcpSocket->state() == QAbstractSocket::UnconnectedState)
        {
            cerr << "PelicanBlobClient: Unable to connect to server ("
                 << _server.toStdString() << ":" << _port << ")" << endl;
            sleep(2);
            continue;
        }

        // Define the data type which the client will except and send request
        StreamDataRequest req;
        DataSpec require;
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
