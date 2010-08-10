#ifndef SUBBAND_SPECTRUM_WIDGET_H_
#define SUBBAND_SPECTRUM_WIDGET_H_

#include <QtGui/QWidget>
#include "pelican/data/DataBlob.h"

#include "pelican/viewer/DataBlobWidget.h"

#include "ui_SubbandSpectraViewer.h"

#include <complex>
#include <vector>

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

class SubbandSpectrumWidget : public DataBlobWidget, public Ui::SubbandSpectraViewer
{
    Q_OBJECT

    public:
        SubbandSpectrumWidget(const ConfigNode& config, QWidget* parent = 0);

        virtual ~SubbandSpectrumWidget() {}

        void updateData(DataBlob* data);

     private:
        void _plot(const std::vector<double>& vec);
        std::vector<double> _spectrumAmp;
        unsigned int _integrationCount;
};

PELICAN_DECLARE(DataBlobWidget, SubbandSpectrumWidget);

} // namespace lofar
} // namespace pelican
#endif // SUBBAND_SPECTRUM_WIDGET_H_
