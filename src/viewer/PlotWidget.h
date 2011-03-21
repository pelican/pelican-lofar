#ifndef PLOT_WIDGET_H_
#define PLOT_WIDGET_H_

#include <QtCore/QObject>

#include <qwt_plot.h>
#include <qwt_plot_curve.h>

class QwtPlotPanner;
class QwtPlotZoomer;
class QwtPlotGrid;

namespace pelican {
namespace lofar {

class PlotPicker;

/**
 * @class PlotWidget
 *
 * @brief
 *
 * @details
 */

class PlotWidget : public  QwtPlot
{
    Q_OBJECT

    public:
        /// Constructor.
        PlotWidget(QWidget* parent = 0);

        /// Destructor.
        virtual ~PlotWidget();

    public:
        /// Plot a curve.
        void plotCurve(unsigned nPoints, const double* xData,
                const double* yData);

        /// Update the base zoom level.
        void updateZoomBase();

    public slots:
        /// Clear the plot canvas.
        void clear();

        /// Set the plot title
        void setPlotTitle(const QString&);

        /// Sets the x axis label
        void setXLabel(const QString&);

        /// Sets the y axis label
        void setYLabel(const QString&);

        /// Show the plot grid.
        void showGrid(bool on);

        /// Show minor grid ticks.
        void showGridMinorTicks(bool);

        /// Export the plot widget as a PNG image.
        void savePNG(QString fileName = "", unsigned sizeX = 500,
                unsigned sizeY = 500);

        /// Save the plot widget as a PDF.
        void savePDF(QString fileName = "", unsigned sizeX = 500,
                unsigned sizeY = 500);

    private:

        /// Set the plot grid object.
        void _setGrid();

        /// Set the plot panning object.
        void _setPanner();

        /// Setup signal and slot connections.
        void _connectSignalsAndSlots();

    signals:
        /// Signal emitted when the plot title changes.
        void titleChanged(const QString&);

        /// Signal emitted when the x axis label changes.
        void xLabelChanged(const QString&);

        /// Signal emitted when the y axis label changes.
        void yLabelChanged(const QString&);

        /// Emitted when grid is enabled.
        void gridEnabled(bool);

        /// Signal for toggle showing minor grid ticks
        void gridMinorTicks(bool);

        /// Signal emitted when the plot is complete.
        void plotComplete();

    private:
        QwtPlotCurve _curve;
        PlotPicker* _picker;
        QwtPlotZoomer* _zoomer;
        QwtPlotGrid* _grid;
        QwtPlotPanner* _panner;
};


} // namespace pelican
} // namespace lofar
#endif // PLOTWIDGET_H_
