#include "viewer/LofarDataViewer.h"
#include "viewer/ThreadedBlobClient.h"


namespace pelican {

namespace lofar {


/**
 *@details LofarDataViewer
 */
LofarDataViewer::LofarDataViewer( const ConfigNode& config, QWidget* parent )
    : DataViewer(config, parent)
{
    _dataStream = "ChannelisedStreamData";
    enableStream(_dataStream);
    QSet<QString> set;
    set.insert(_dataStream);
    _updatedStreams( set );
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
