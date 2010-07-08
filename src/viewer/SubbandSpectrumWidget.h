#ifndef SUBBAND_SPECTRUM_WIDGET_H_
#define SUBBAND_SPECTRUM_WIDGET_H_

#include <QtGui/QWidget>
#include "pelican/data/DataBlob.h"
#include "DataBlobWidget.h"

#include "ui_SubbandSpectraViewer.h"

/**
 * @file SubbandSpectrumWidget.h
 */

namespace pelican {
namespace lofar {

/**
 * @class SubbandSpectrumWidget
 *
 * @brief

 * @details
 */

class SubbandSpectrumWidget : public DataBlobWidget, public Ui::SubbandSpectraViewer
{
    Q_OBJECT

    public:
        SubbandSpectrumWidget(QWidget* parent = 0)
        : DataBlobWidget(parent)
        {
            setupUi(this);
        }
        virtual ~SubbandSpectrumWidget() {}

        void updateData(DataBlob* data) {
            //unsigned s = spingBox_lldas=>value
            //float* yData = data->ptr(s, p);
            //PlotWidget_spectrumView->plotCurve(xData, yData, nPoints);
        }

    private:
};

} // namespace lofar
} // namespace pelican
#endif // SUBBAND_SPECTRUM_WIDGET_H_
