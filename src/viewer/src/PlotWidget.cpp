#include "viewer/PlotWidget.h"
#include "viewer/PlotPicker.h"

#include <qwt-qt4/qwt_plot_zoomer.h>
#include <qwt-qt4/qwt_plot_grid.h>
#include <qwt-qt4/qwt_symbol.h>
#include <qwt-qt4/qwt_plot_panner.h>
#include <qwt-qt4/qwt_plot_layout.h>


#include <QtGui/QColor>
#include <QtGui/QFileDialog>
#include <QtCore/QFileInfo>
#include <QtGui/QPrinter>

namespace pelican {
namespace lofar {

/**
 * @details
 * @param parent
 */
PlotWidget::PlotWidget(QWidget* parent) : QwtPlot(parent)
{
    // Setup the plot grid object.
    _setGrid();
    showGrid(false);
    showGridMinorTicks(false);

    // Setup the plot panner object.
    _setPanner();

    // Setup the plot selection object.
    _picker = new PlotPicker(QwtPlot::xBottom, QwtPlot::yLeft, canvas());

    // Setup the plot zoomer object.
    _zoomer = new QwtPlotZoomer(QwtPlot::xBottom, QwtPlot::yLeft,
            canvas());
    _zoomer->setTrackerMode(QwtPlotZoomer::ActiveOnly);
    _zoomer->setRubberBandPen(QColor(Qt::green));
    _zoomer->setTrackerPen(QPen(QColor(Qt::green)));
    _zoomer->setMousePattern(QwtEventPattern::MouseSelect1, Qt::LeftButton);
    _zoomer->setMousePattern(QwtEventPattern::MouseSelect3, Qt::RightButton,
            Qt::MetaModifier);
    _zoomer->setMousePattern(QwtEventPattern::MouseSelect2, Qt::RightButton);

    // Set the curve style.
    QwtSymbol symbol;
    symbol.setStyle(QwtSymbol::Cross);
    symbol.setPen(QPen(QColor(Qt::black)));
    symbol.setBrush(QBrush(QColor(Qt::black)));
    symbol.setSize(5);
    _curve.setSymbol(symbol);
    _curve.setStyle(QwtPlotCurve::NoCurve);


    _connectSignalsAndSlots();

    // Clear the plot widget setting title and axis defaults.
    clear();
}


/**
 * @details
 */
PlotWidget::~PlotWidget()
{
    delete _picker;
    delete _zoomer;
    delete _grid;
    delete _panner;
}


/**
 * @details
 * Plot a curve.
 *
 * @param nPoints	Number of points to plot (length of x and y data arrays)
 * @param xData		x Data array.
 * @param yData		y Data array.
 */
void PlotWidget::plotCurve(unsigned nPoints, const double* xData,
        const double* yData)
{
    if (nPoints == 0 || xData == NULL || yData == NULL) {
        throw QString("PlotWidget::plotCurve(): Input data error.");
    }

    _curve.setData(xData, yData, nPoints);

    setAxisAutoScale(QwtPlot::yLeft);
    setAxisAutoScale(QwtPlot::xBottom);
    _updateZoomBase();

    //_curve.attach(this);
    replot();

    emit(plotComplete());
}


/**
 * @details
 *
 */
void PlotWidget::clear()
{
    setPlotTitle("");
    setXLabel("");
    setYLabel("");

    setAxisScale(QwtPlot::yLeft, 0.0, 1.0);
    setAxisScale(QwtPlot::xBottom, 0.0, 1.0);
    enableAxis(QwtPlot::yRight, false);
    plotLayout()->setAlignCanvasToScales(true);

    //_curve.detach();

    replot();
}



/**
 * @details
 */
void PlotWidget::setPlotTitle(const QString& text)
{
    QFont f;
    f.setPointSize(12);
    QwtText title(text);
    title.setFont(f);
    setTitle(title);
    emit(titleChanged(text));
}


/**
 * @details
 */
void PlotWidget::setXLabel(const QString& text)
{
    QFont f;
    f.setPointSize(10);
    QwtText label(text);
    label.setFont(f);
    setAxisTitle(QwtPlot::xBottom, label);
    emit(xLabelChanged(text));
}


/**
 * @details
 */
void PlotWidget::setYLabel(const QString& text)
{
    QFont f;
    f.setPointSize(10);
    QwtText label(text);
    label.setFont(f);
    setAxisTitle(QwtPlot::yLeft, label);
    emit(yLabelChanged(text));
}



/**
 * @details
 *
 * @param on
 */
void PlotWidget::showGrid(bool on)
{
    _grid->setVisible(on);
    replot();
    emit(gridEnabled(on));
}


/**
 * @details
 *
 * @param on
 */
void PlotWidget::showGridMinorTicks(bool on)
{
    _grid->enableXMin(on);
    _grid->enableYMin(on);
    emit(gridMinorTicks(on));
}


/**
 * @details
 */
void PlotWidget::savePNG(QString fileName, unsigned sizeX, unsigned sizeY)
{
    if (fileName.isEmpty()) {
        fileName = QFileDialog::getSaveFileName(this,
                "Save plot widget: file name", QString(),
                "(*.png)");
    }

    if (fileName.isEmpty()) {
        return;
    }

    QFileInfo fileInfo(fileName);
    if (fileInfo.suffix() != "png") {
        fileName += ".png";
    }

    QPixmap pixmap(sizeX, sizeY);
    pixmap.fill();
    print(pixmap);
    int quality = -1; // -1 = default, [0..100] otherwise.
    if (!pixmap.save(fileName, "PNG", quality)) {
        throw QString("PlotWidget::exportPNG(): Error saving PNG");
    }
}


/**
 *
 * @param fileName
 */
void PlotWidget::savePDF(QString fileName, unsigned sizeX, unsigned sizeY)
{
    if (fileName.isEmpty()) {
        fileName = QFileDialog::getSaveFileName(this,
                "Save plot widget: file name", QString(),
                "(*.pdf)");
    }

    if (fileName.isEmpty()) {
        return;
    }

    QFileInfo fileInfo(fileName);
    if (fileInfo.suffix() != "pdf") {
        fileName += ".pdf";
    }

    //QPrinter printer(QPrinter::HighResolution);
    QPrinter printer(QPrinter::ScreenResolution);
    printer.setDocName(fileName);
    printer.setOutputFileName(fileName);
    printer.setCreator("wfit");
    printer.setColorMode(QPrinter::Color);
    printer.setOutputFormat(QPrinter::PdfFormat);
    //printer.setPaperSize(QPrinter::A4);
    printer.setPaperSize(QSizeF(sizeX, sizeY), QPrinter::Point);
    print(printer);

}



/**
 * @details
 */
void PlotWidget::_updateZoomBase()
{
    _zoomer->setZoomBase(true);
}


/**
 * @details
 */
void PlotWidget::_setGrid()
{
    _grid = new QwtPlotGrid();
    _grid->attach(this);
    _grid->enableXMin(this);
    _grid->enableYMin(this);
    _grid->setMinPen(QPen(QBrush(QColor(Qt::lightGray)), qreal(0.0), Qt::DotLine));
    _grid->setMajPen(QPen(QBrush(QColor(Qt::darkGray)),  qreal(0.0), Qt::DotLine));
}


/**
 * @details
 */
void PlotWidget::_setPanner()
{
    _panner = new QwtPlotPanner(this->canvas());
    _panner->setMouseButton(Qt::MidButton);
    _panner->setAxisEnabled(QwtPlot::yRight, false);
}


/**
 * @details
 * Sets signal and slot connections.
 */
void PlotWidget::_connectSignalsAndSlots()
{
}


} // namespace pelican
} // namespace lofar
