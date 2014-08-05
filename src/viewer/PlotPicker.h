#ifndef PLOT_PICKER_H_
#define PLOT_PICKER_H_

/**
 * @file PlotPicker.h
 */

#include <qwt_plot_picker.h>
#include <qwt_text.h>

namespace pelican {
namespace ampp {

/**
 * @class PlotPicker
 *
 * @brief
 *
 * @details
 */

class PlotPicker : public QwtPlotPicker
{
    public:
        enum { POINT_SELECT, RECT_SELECT };

    public:
        /// Constructs a plot picker object.
        PlotPicker(int xAxis, int yAxis, QwtPlotCanvas* canvas);

        /// Destroys the plot picker
        virtual ~PlotPicker() {}

    public:
        /// Sets the selection type
        void setSelectionType(unsigned type = POINT_SELECT);

    protected:
        /// Returns the tracker text.
        virtual QwtText trackerText(const QPoint& point) const;
        virtual QwtText trackerText(const QwtDoublePoint& pos) const;
};

} // namespace ampp
} // namespace pelican
#endif // PLOT_PICKER_H_
