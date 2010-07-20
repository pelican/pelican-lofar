#include "viewer/LofarDataViewer.h"
#include "viewer/ThreadedBlobClient.h"

#include "pelican/utility/ConfigNode.h"

#include <QtCore/QDebug>
#include <iostream>

namespace pelican {

namespace lofar {


/**
 *@details LofarDataViewer
 */
LofarDataViewer::LofarDataViewer(const ConfigNode& config, QWidget* parent)
: DataViewer(config, parent)
{
    // Get configuration options.
    _dataStream = config.getOption("dataStream", "name", "SubbandSpectraStokes");
    _port = (quint16)config.getOption("server", "port", "6969").toUInt();
    _address = config.getOption("server", "address", "127.0.0.1");

//    std::cout << "port = " << _port << std::endl;
//    std::cout << "address = " << _address.toStdString() << std::endl;
//    std::cout << "stream = " << _dataStream.toStdString() << std::endl;

    // Set the data stream name and activate it with the data viewer.
    enableStream(_dataStream);

    // Update the Gui for the specified streams.
    QSet<QString> set;
    set.insert(_dataStream);
    _updatedStreams(set);

//    // TODO make sure some ports and server get set here... (check Alessio's code)
//    // with the right config these come from the DataViewer constructor
//    // pulling them out of the config.
//    qDebug() << "FIXME: LofarDataViewer(): server = " << server() << endl;
//    qDebug() << "FIXME: LofarDataViewer(): port = " << port() << endl;

    //_client = new ThreadedBlobClient( server(), port(), _dataStream);
    _client = new ThreadedBlobClient(_address, _port, _dataStream);

    connect(_client, SIGNAL(dataUpdated(const QString& , DataBlob*)),
            this, SLOT(dataUpdated(const QString& , DataBlob*)));
}

/**
 *@details
 */
LofarDataViewer::~LofarDataViewer()
{
    delete _client;
}

} // namespace lofar
} // namespace pelican
