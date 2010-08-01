#include "viewer/LofarDataViewer.h"
#include "viewer/ThreadedBlobClient.h"
#include "viewer/SubbandSpectrumWidget.h"

#include "pelican/utility/Config.h"
#include "pelican/utility/ConfigNode.h"

#include <QtCore/QDebug>
#include <iostream>

namespace pelican {

namespace lofar {


/**
 *@details LofarDataViewer
 */
LofarDataViewer::LofarDataViewer(const Config& config, const Config::TreeAddress& base, QWidget* parent)
: DataViewer(config, base, parent)
{
    // Get configuration options.
    ConfigNode configN = config.get(base);
    _dataStream = configN.getOption("dataStream", "name", "SubbandSpectraStokes");
    _port = (quint16)configN.getOption("server", "port", "6969").toUInt();
    _address = configN.getOption("server", "address", "127.0.0.1");

    setStreamViewer("SubbandSpectraStokes","SubbandSpectrumWidget");
//    std::cout << "port = " << _port << std::endl;
//    std::cout << "address = " << _address.toStdString() << std::endl;
//    std::cout << "stream = " << _dataStream.toStdString() << std::endl;

    // Set the data stream name and activate it with the data viewer.
    enableStream(_dataStream);

    // Set the viewer

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
