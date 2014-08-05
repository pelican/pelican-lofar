#ifndef FILEDATAVIEWER_H
#define FILEDATAVIEWER_H

#include <QWidget>
#include <QVector>
#include <QMap>
class QPushButton;
class QVBoxLayout;
#include "pelican/utility/FactoryConfig.h"
#include "pelican/utility/FactoryGeneric.h"
#include "pelican/data/DataBlob.h"
#include <string>

/**
 * @file FileDataViewer.h
 */

namespace pelican {
class DataBlobFileReader;
class DataBlobWidget;
class DataBlobWidgetFactory;
class Config;

namespace ampp {

/**
 * @class FileDataViewer
 *  
 * @brief
 *     Widget for viewing DataBlob files
 * @details
 * 
 */

class FileDataViewer : public QWidget
{
    Q_OBJECT

    public:
        FileDataViewer( Config* config, QWidget* parent=0 );
        ~FileDataViewer();
        void addFiles( const QVector<std::string>& files );
        void setDataBlobViewer( const QString& blobType, const QString& viewerType );

    protected:
        FactoryGeneric<DataBlob>* dataBlobFactory();
        DataBlobWidget* getDataBlobViewer(const QString& stream) const;
        void error( const QString& msg ) const;
        
    private slots:
        // get the next DataBlob
        void next();

    private:
        QVector<std::string> _files;
        QPushButton* _next;
        DataBlobFileReader* _currentFile;
        DataBlob* _blob;
        QString _currentType;
        QString _nextType;
        DataBlobWidgetFactory* _viewerFactory;
        QWidget* _viewerWidget;
        QVBoxLayout* _layout;
        int _viewerPos; // position of data display widget in the layout
        QMap<QString,QString> _viewerMap;
};

} // namespace ampp
} // namespace pelican
#endif // FILEDATAVIEWER_H 
