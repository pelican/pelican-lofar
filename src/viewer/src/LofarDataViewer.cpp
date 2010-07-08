#include "viewer/LofarDataViewer.h"
#include "lib/PelicanBlobClient.h"
#include "lib/ChannelisedStreamData.h"


namespace pelican {

namespace lofar {


/**
 *@details LofarDataViewer 
 */
LofarDataViewer::LofarDataViewer( const ConfigNode& config, QWidget* parent )
    : DataViewer(config, parent)
{
    enableStream(_dataStream);
    QSet<QString> set;
    set.insert(_dataStream);
    _updatedStreams( set );
    //_client = new PelicanBlobClient( _dataStream, server(), port() );
    //run();
}

/**
 *@details
 */
LofarDataViewer::~LofarDataViewer()
{
}

void LofarDataViewer::run()
{
    ChannelisedStreamData blob;
    while( 1 )
    {
        QHash<QString, DataBlob*> dataHash;
        dataHash.insert(_dataStream, &blob);
        _client->getData(dataHash);
        dataUpdated(_dataStream, &blob);
    }
}

} // namespace lofar
} // namespace pelican
