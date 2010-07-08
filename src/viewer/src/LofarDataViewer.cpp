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
    _dataStream = "ChannelisedStreamData";
    enableStream(_dataStream);
    QSet<QString> set;
    set.insert(_dataStream);
    _updatedStreams(set);


    // TODO make sure some ports and server get set here... (check Alessio's code)
    qDebug() << "LofarDataViewer(): server = " << server() << endl;
    qDebug() << "LofarDataViewer(): port = " << port() << endl;

    _client = new ThreadedBlobClient( server(), port(), _dataStream);
    connect ( _client, SIGNAL(dataUpdated(const QString& , DataBlob* ) ),
              this, SLOT(dataUpdated(const QString& , DataBlob*)) );
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
