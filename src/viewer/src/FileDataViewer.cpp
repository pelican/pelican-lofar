#include "viewer/FileDataViewer.h"
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QMessageBox>
#include <QLabel>
#include "pelican/utility/Config.h"
#include "pelican/output/DataBlobFileReader.h"
#include "pelican/viewer/DataBlobWidgetFactory.h"
#include <iostream>

namespace pelican {

namespace ampp {


/**
 *@details FileDataViewer 
 */
FileDataViewer::FileDataViewer(Config* config, QWidget* parent)
    : QWidget(parent), _currentFile(0), _blob(0), _viewerWidget(0)
{
    Config::TreeAddress widgetNodeAddress;
    _viewerFactory = new DataBlobWidgetFactory( config, widgetNodeAddress );

    // ----- Setup the GUI interfaca ----------
    QWidget *bottomFiller = new QWidget;
    bottomFiller->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    // Add scrolling controls
    QWidget* commandBar = new QWidget;
    QHBoxLayout* cblayout = new QHBoxLayout;
    _next = new QPushButton("Next");
    _next->setEnabled( false );
    connect( _next, SIGNAL( clicked() ), this, SLOT(next()) );
    cblayout->addWidget(_next);
    commandBar->setLayout(cblayout);

    _layout = new QVBoxLayout;
    _viewerWidget = new QLabel("File Data Viewer");
    _viewerWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    static_cast<QLabel*>(_viewerWidget)->setFrameStyle(QFrame::Panel | QFrame::Raised);
    _layout->addWidget( _viewerWidget );
    _viewerPos = _layout->count() - 1;
    _layout->addWidget( commandBar );
    //layout->addWidget( bottomFiller );

    setLayout(_layout);

}

/**
 *@details
 */
FileDataViewer::~FileDataViewer()
{
     delete _viewerFactory;
}

void FileDataViewer::addFiles( const QVector<std::string>& files ) {
     if( files.size() ) {
        _files += files;
        _next->setEnabled( true );
     }
}

void FileDataViewer::next() {
     // open the next file if existing file is exhausted
     while( ! _currentFile ) {
         if( _files.size() <= 0 ) {
            _next->setEnabled( false );
            return;
         }
         QString filename = QString(_files.first().c_str());
         _files.pop_front();
         _currentFile = new DataBlobFileReader;
         try {
             _currentFile->open(filename);
         }
         catch( ... ) {
            delete _currentFile;
            _currentFile = 0;
            error(QString("FileDataViewer::next(): Cannot open file %1").arg(filename));
         }
     }
     // -- get the next blob from the file
     if( _nextType == "" ) {
        _nextType = _currentFile->nextBlob();
     }
     if( _nextType != "" ) {
         // check there is any data left
         delete _blob; // delete the previous blob
         _blob = dataBlobFactory()->create(_nextType);
         _currentFile->readData( _blob );

         // -- get a suitable viewer widget
         if( _currentType != _nextType ) {
             _layout->removeWidget( _viewerWidget );
             delete _viewerWidget;
             _viewerWidget = getDataBlobViewer(_nextType);
             _layout->insertWidget( _viewerPos, _viewerWidget );
             _currentType = _nextType;
         }
         static_cast<DataBlobWidget*>(_viewerWidget)->updateData( _blob );
         _nextType = _currentFile->nextBlob();
     }
     if( _nextType == "" ) {
        delete _currentFile;
        _currentFile = 0;
        if( _files.size() <= 0 ) 
            _next->setEnabled( false );
        return;
     }

}

void FileDataViewer::error( const QString& msg ) const {
    QMessageBox msgBox;
    msgBox.setText(msg);
    msgBox.exec();
}

void FileDataViewer::setDataBlobViewer( const QString& stream, const QString& viewerType)
{
    _viewerMap[stream] = viewerType;
}

DataBlobWidget* FileDataViewer::getDataBlobViewer(const QString& stream) const
{
    DataBlobWidget* widget = (DataBlobWidget*)0;
    try {
        if( _viewerMap.contains(stream) ) {
            widget = _viewerFactory->create( _viewerMap[stream] );
        }
        else
        {
            widget = _viewerFactory->create( stream + "Widget" );
        }
    }
    catch( const QString& e )
    {
        // default is the basic datablob information Widget
        error(QString("Problem instatiating a viewer for data of type \"%1\" : %2").arg(stream).arg(e));
        widget = _viewerFactory->create( "DataBlobWidget" );
    }
    return widget;
}

FactoryGeneric<DataBlob>* FileDataViewer::dataBlobFactory()
{
    static FactoryGeneric<DataBlob> factory(false);
    return &factory;
}

} // namespace ampp
} // namespace pelican
