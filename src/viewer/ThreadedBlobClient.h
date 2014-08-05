#ifndef THREADEDBLOBCLIENT_H
#define THREADEDBLOBCLIENT_H


#include <QtCore/QThread>

/**
 * @file ThreadedBlobClient.h
 */

namespace pelican {
    class DataBlob;

namespace ampp {
    class PelicanBlobClient;

/**
 * @class ThreadedBlobClient
 *
 * @brief
 *   Wraps the PelicanBlobServer in a seperate thread
 * @details
 *   Allows to emit messages when data arrives etc.
 */

class ThreadedBlobClient : public QThread
{
    Q_OBJECT

    public:
        ThreadedBlobClient( const QString& host, quint16 port, const QString& stream, QObject* parent  = 0 );
        ~ThreadedBlobClient();

    protected:
        virtual void run();

    signals:
        void dataUpdated(const QString& streamName, DataBlob* );

    private:
        PelicanBlobClient* _client;
        QString _host;
        quint16 _port;
        QString _dataStream; // single stream only
        bool _isRunning;
};

} // namespace ampp
} // namespace pelican
#endif // THREADEDBLOBCLIENT_H
