#ifndef PELICANBLOBCLIENT_H
#define PELICANBLOBCLIENT_H

// Pelican stuff
#include "pelican/comms/AbstractClientProtocol.h"

// Qt stuff
#include <QtNetwork/QTcpSocket>
#include <QtCore/QString>

/**
 * @file PelicanBlobClient.h
 */

namespace pelican {
namespace lofar {

/**
 * @class PelicanBlobClient
 *
 * @ingroup pelican_lofar
 *
 * @brief
 * Implements the data client interface for attaching to a Pelican TCP Server
 *
 * @details
 *
 */

class PelicanBlobClient
{
    public:
        PelicanBlobClient(QString blobType, QString server, quint16 port);
        ~PelicanBlobClient();

    public:
        void getData(QHash<QString, DataBlob*>& dataHash);

    private:
        void connectToLofarPelican();

    private:
        AbstractClientProtocol* _protocol;
        QTcpSocket* _tcpSocket;
        QString _server;
        QString _blobType;
        unsigned _port;
};

} // namespace lofar
} // namespace pelican

#endif // PELICANBLOBCLIENT_H
