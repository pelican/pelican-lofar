#include "viewer/PlotPicker.h"
#include <QtCore/QString>

namespace pelican {
namespace lofar {


/**
 * @details
 * Constructor for Picker object. Sets various defaults.
 */
PlotPicker::PlotPicker(int xAxis, int yAxis, QwtPlotCanvas* canvas) :
    QwtPlotPicker(xAxis, yAxis, canvas)
{
    setSelectionType(POINT_SELECT);
    setMousePattern(QwtEventPattern::MouseSelect1, Qt::LeftButton, Qt::ShiftModifier);
    setRubberBandPen(QPen(QColor(Qt::green)));
    setTrackerPen(QPen(QColor(Qt::darkBlue)));
    setTrackerMode(QwtPicker::ActiveOnly);
}


/**
 * @details
 */
void PlotPicker::setSelectionType(unsigned type)
{
    if (type == RECT_SELECT) {
        setSelectionFlags(QwtPicker::RectSelection | QwtPicker::DragSelection);
        setRubberBand(QwtPicker::RectRubberBand);
    }
    else if (type == POINT_SELECT) {
        setSelectionFlags(QwtPicker::PointSelection | QwtPicker::DragSelection);
        setRubberBand(QwtPicker::CrossRubberBand);
    }
    else {
        throw QString("PlotPicker::setSelectionType(): Unknown selection type.");
    }
}


/**
* @details
* Function to translate a pixel position into a string
*/
QwtText PlotPicker::trackerText(const QPoint& pos) const
{
    QColor colour(Qt::white);
    colour.setAlpha(230);
    QwtText text = QString::number(pos.x(),'f',4) + ", " +
                            QString::number(pos.y(),'f',4);
    text.setBackgroundBrush(QBrush(colour));
    return text;
}


/**
* @details
* Function to translate a pixel position into a string
*/
QwtText PlotPicker::trackerText(const QwtDoublePoint& pos) const
{
    return trackerText(pos);
}


} // namespace lofar
} // namespace pelican
