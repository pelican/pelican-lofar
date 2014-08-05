#ifndef LOFARDATAVIEWER_H
#define LOFARDATAVIEWER_H
#include <QtCore/QSet>


#include "pelican/viewer/DataViewer.h"

/**
 * @file LofarDataViewer.h
 */

namespace pelican {

namespace ampp {

class ThreadedBlobClient;

/**
 * @class LofarDataViewer
 *
 * @brief
 *    A lofar specific single stream viewer
 *
 * @details
 *
 */

class LofarDataViewer : public DataViewer
{
    Q_OBJECT

    public:
        LofarDataViewer(const Config& config, const Config::TreeAddress& base,
                QWidget* parent=0);
        ~LofarDataViewer();

    private:
        ThreadedBlobClient* _client;
        QString _dataStream;
        quint16 _port;
        QString _address;
};

} // namespace ampp
} // namespace pelican
#endif // LOFARDATAVIEWER_H
