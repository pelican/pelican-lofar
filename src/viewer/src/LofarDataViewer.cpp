#include "viewer/LofarDataViewer.h"
#include "viewer/ThreadedBlobClient.h"
#include "viewer/SpectrumDataSetWidget.h"

#include "pelican/utility/Config.h"
#include "pelican/utility/ConfigNode.h"

#include <QtCore/QDebug>
#include <iostream>

namespace pelican {
namespace lofar {


/**
 * @details LofarDataViewer
 */
LofarDataViewer::LofarDataViewer(const Config& config,
        const Config::TreeAddress& base, QWidget* parent)
: DataViewer(config, base, parent)
{
    // Get configuration options.
    ConfigNode configN = config.get(base);
    _dataStream = configN.getOption("dataStream", "name", "SpectrumDataSetStokes");
    _port = (quint16)configN.getOption("server", "port", "6969").toUInt();
    _address = configN.getOption("server", "address", "127.0.0.1");

    // Set the viewer
    setStreamViewer("SpectrumDataSetStokes", "SpectrumDataSetWidget");

    // Set the data stream name and activate it with the data viewer.
    enableStream(_dataStream);

    // Update the Gui for the specified streams.
    QSet<QString> set;
    set.insert(_dataStream);
    _updatedStreams(set);

    _client = new ThreadedBlobClient(_address, _port, _dataStream);

    connect(_client, SIGNAL(dataUpdated(const QString& , DataBlob*)),
            this, SLOT(dataUpdated(const QString& , DataBlob*)));
}

/**
 * @details
 */
LofarDataViewer::~LofarDataViewer()
{
    delete _client;
}

} // namespace lofar
} // namespace pelican
