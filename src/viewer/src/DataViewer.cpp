#include "viewer/DataViewer.h"
#include <QtGui/QMessageBox>
#include <QtGui/QVBoxLayout>
#include <QtGui/QActionGroup>
#include <QtGui/QAction>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QTabWidget>
#include "viewer/DataBlobWidget.h"
#include "pelican/utility/ConfigNode.h"


namespace pelican {

namespace lofar {


/**
 *@details DataViewer 
 *    In development : ideally will use a generic Blob client and be exported to 
 *    pelican
 */
DataViewer::DataViewer(const ConfigNode& config, QWidget* parent)
    : QWidget(parent), _client(0)
{
    // setup generic widgets
    QWidget *topFiller = new QWidget;
    topFiller->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    QWidget *bottomFiller = new QWidget;
    bottomFiller->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    _streamTabs = new QTabWidget;

    // setup actions
    QMenuBar* menubar = new QMenuBar;

    // Main Menu
    QMenu* mainMenu = new QMenu(tr("Main"));
    QAction* exitAct = new QAction(tr("E&xit"), this);
    exitAct->setShortcut(tr("Ctrl+Q"));
    exitAct->setStatusTip(tr("Exit the application"));
    connect(exitAct, SIGNAL(triggered()), this, SLOT(close()));
    mainMenu->addAction(exitAct);
    menubar->addMenu(mainMenu);

    // View Menu
    _viewMenu = new QMenu(tr("View"));
    _viewMenu->setTearOffEnabled(true);
    menubar->addMenu(_viewMenu);
    _streamActionGroup = new QActionGroup(this);
    _streamActionGroup->setExclusive(false);
    _viewMenu->addSeparator();

    // Help Menu
    QMenu* helpMenu = new QMenu(tr("Help"));
    QAction* aboutAct = new QAction(tr("&About"), this);
    connect(aboutAct, SIGNAL(triggered()), this, SLOT(about()));
    helpMenu->addAction(aboutAct);
    menubar->addMenu(helpMenu);

    // set the layout
    QVBoxLayout* layout = new QVBoxLayout;
    layout->setMargin(5);
    layout->addWidget(menubar);
    layout->addWidget(topFiller);
    layout->addWidget(_streamTabs);
    layout->addWidget(bottomFiller);
    setLayout(layout);

    setConfig(config);
}

/**
 *@details
 */
DataViewer::~DataViewer()
{
    delete _streamTabs;
    delete _viewMenu;
    delete _streamActionGroup;
    delete _streamTabs;
}

DataBlobWidget* DataViewer::getWidgetViewer(const QString& stream) const
{
    DataBlobWidget* widget = (DataBlobWidget*)0;
    if( _streamViewerMap.contains(stream) ) {
        // TODO -- factory to create Viewers of the appropriate type
        //widget = _viewerfactory->create( _streamViewerMap[stream], stream );
    }
    else {
        widget = new DataBlobWidget;
    }
    return widget;
}

void DataViewer::setConfig(const ConfigNode& config)
{
    // set basic connection information
    _port = config.getOption("connection", "port").toInt();
    _server = config.getOption("connection", "host");

    // set stream activation defaults
    // read in from the configuration TODO
    // _defaultsEnabled[stream] = true;

}

/*
 * @details
 * updates the Gui to reflect the specified streams
 */
void DataViewer::_updatedStreams( const QSet<QString>& streams )
{
    // clean up the previous state
    foreach(QAction* action, _streamActionGroup->actions() ) 
    {
        _streamActionGroup->removeAction(action);
        _viewMenu->removeAction(action);
        delete action;
    }

    // set up the new streams
    foreach(const QString& stream, streams )
    {
        QAction* a = _streamActionGroup->addAction(stream);
        a->setCheckable(true);
        _viewMenu->addAction(a);
        connect( a, SIGNAL(triggered() ), this, SLOT(_streamToggled()) );
        if( _defaultsEnabled.contains(stream) )
        {
            a->setChecked(true);
            enableStream(stream);
        }
        else {
            _defaultsEnabled[stream] = false;
            a->setChecked(false);
        }

    }
}

void DataViewer::about()
{
    QMessageBox::about(this, tr("Data Stream Viewer"),
            tr("Connects to Pelican data Streams "
                "and displays them."));
}

void DataViewer::connectStreams()
{
    // construct a request for all the streams required
    // TODO
}

void DataViewer::dataUpdated(const QString& stream, DataBlob* blob)
{
    if(  _activeStreams.contains(stream) )
        static_cast<DataBlobWidget*>(_streamTabs->widget(_activeStreams[stream]))->updateData(blob);
}

void DataViewer::enableStream( const QString& stream )
{
    if( ! _activeStreams.contains(stream) )
    {
        _defaultsEnabled[stream] = true;
        _activeStreams[stream] = _streamTabs->addTab( getWidgetViewer(stream), stream );
    }
}

void DataViewer::disableStream( const QString& stream )
{
    if(  _activeStreams.contains(stream) )
    {
        _streamTabs->removeTab( _activeStreams[stream] );
        _activeStreams.remove(stream);
        _defaultsEnabled[stream] = false;
    }
}

bool DataViewer::toggleStream( const QString& stream )
{
    if( _activeStreams.contains(stream) )
    {
        disableStream(stream);
        return false;
    }
    else {
        enableStream(stream);
        return true;
    }
}

void DataViewer::_streamToggled()
{
    QAction* action = static_cast<QAction*>(sender());
    if( action ) 
        toggleStream( action->text() );
}

} // namespace lofar
} // namespace pelican
