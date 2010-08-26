#ifndef SPECTRUM_DATA_SET_WIDGET_H_
#define SPECTRUM_DATA_SET_WIDGET_H_

#include <QtGui/QWidget>
#include "pelican/data/DataBlob.h"

#include "pelican/viewer/DataBlobWidget.h"

#include "ui_SpectrumDataSetViewer.h"

#include <complex>
#include <vector>

using std::vector;

/**
 * @file SubbandSpectrumWidget.h
 */

namespace pelican {

class ConfigNode;

namespace lofar {

/**
 * @class SubbandSpectrumWidget
 *
 * @brief

 * @details
 */

class SpectrumDataSetWidget : public DataBlobWidget, public Ui::SpectrumDataSetViewer
{
    Q_OBJECT

    public:
        SpectrumDataSetWidget(const ConfigNode& config, QWidget* parent = 0);
        virtual ~SpectrumDataSetWidget() {}

        void updateData(DataBlob* data);

    private:
        void _plot(const vector<double>& vec);

        vector<double> _spectrumAmp;
        unsigned _integrationCount;
};

PELICAN_DECLARE(DataBlobWidget, SpectrumDataSetWidget)

} // namespace lofar
} // namespace pelican
#endif // SPECTRUM_DATA_SET_WIDGET_H_
