#ifndef DATAVIEWER_H
#define DATAVIEWER_H

#include <QtGui/QWidget>
#include <QtCore/QMap>
#include <QtCore/QString>
#include <QtCore/QSet>
class QActionGroup;
class QTabWidget;
class QMenu;

/**
 * @file DataViewer.h
 */

namespace pelican {
    class ConfigNode;
    class DataBlobClient;
    class DataBlob;

namespace lofar {
    class DataBlobWidget;

/**
 * @class DataViewer
 *
 * @brief
 *    base class to view data streams
 * @details
 *
 */

class DataViewer : public QWidget
{
    Q_OBJECT

    public:
        DataViewer( const ConfigNode& config, QWidget* parent=0 );
        ~DataViewer();

        // returns the connected server host
        const QString& server() { return _server; };

        // returns the connected server port
        quint16 port() { return _port; };

        // enable tracking of the specified stream
        void enableStream( const QString& );
        // disable tracking of the specified stream
        void disableStream( const QString& );
        // flip the tracking status of the specifed stream
        // returns true if the new status is on false if off
        bool toggleStream( const QString& );

        // set internal state from the configuration file
        void setConfig( const ConfigNode& config);

    public slots:
        void dataUpdated(const QString& stream, DataBlob*);
        void about();

    protected:
        virtual DataBlobWidget* getWidgetViewer(const QString& stream) const;
        void connectStreams( );
        void _updatedStreams( const QSet<QString>& streams );
        void closeEvent(QCloseEvent* event);

    protected slots:
        void _streamToggled();

    private:
        // GUI members
        QTabWidget* _streamTabs;
        QMenu* _viewMenu;
        QActionGroup* _streamActionGroup;

        DataBlobClient* _client;

        QMap<QString,int> _activeStreams; // currently activated streams and their tab id
        QMap<QString,bool> _defaultsEnabled; // the required state of a stream

        quint16 _port;
        QString _server;

        QMap<QString,QString> _streamViewerMap; // maps a stream name on to a type of viewer
};

} // namespace lofar
} // namespace pelican
#endif // DATAVIEWER_H
