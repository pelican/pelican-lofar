#include "viewer/LofarDataViewer.h"
#include "viewer/ThreadedBlobClient.h"

#include <QtCore/QDebug>

namespace pelican {

namespace lofar {


/**
 *@details LofarDataViewer
 */
LofarDataViewer::LofarDataViewer(const ConfigNode& config, QWidget* parent)
: DataViewer(config, parent)
{
    // Set the data stream name and activate it with the data viewer.
    _dataStream = "ChannelisedStreamData";
    enableStream(_dataStream);

    // Update the Gui for the specified streams.
    QSet<QString> set;
    set.insert(_dataStream);
    _updatedStreams(set);


    // TODO make sure some ports and server get set here... (check Alessio's code)
    // with the right config these come from the DataViewer constructor
    // pulling them out of the config.
    qDebug() << "LofarDataViewer(): server = " << server() << endl;
    qDebug() << "LofarDataViewer(): port = " << port() << endl;

    //_client = new ThreadedBlobClient( server(), port(), _dataStream);
    _client = new ThreadedBlobClient("127.0.0.1", qint16(6969), _dataStream);

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
