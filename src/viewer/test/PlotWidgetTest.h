#ifndef PLOT_WIDGET_TEST_H_
#define PLOT_WIDGET_TEST_H_

#include "viewer/PlotWidget.h"
#include "viewer/SpectrumDataSetWidget.h"
#include "pelican/utility/ConfigNode.h"

#include <QtGui/QApplication>
#include <QtCore/QObject>
#include <QtTest/QtTest>

#include <vector>

namespace pelican {
namespace ampp {

class PlotWidgetTest : public QObject
{
    Q_OBJECT

    public:
        PlotWidgetTest() {
            _plotWidget = new PlotWidget;
            _sViewer = new SpectrumDataSetWidget(_config);
        }
        ~PlotWidgetTest() {
            delete _plotWidget;
            delete _sViewer;
        }

    private slots:

        void testPlotWidget() {
            _plotWidget->show();
            _plotWidget->resize(500, 500);

            // Create nPoints data values
            unsigned nPoints = 1000;
            double nPeriods = 2;
            std::vector<float> xData(nPoints);
            std::vector<float> yData(nPoints);

            for (unsigned i = 0; i < nPoints; ++i) {
                xData[i] = float(i);
                double arg = 2 * M_PI * nPeriods/float(nPoints) * xData[i];
                yData[i] = float(sin(arg));
            }

            std::vector<double> xPlot(xData.begin(), xData.end());
            std::vector<double> yPlot(yData.begin(), yData.end());

            _plotWidget->plotCurve(nPoints, &xPlot[0], &yPlot[0]);
        }

        void testSubbandSpectrumWidget()
        {
            _sViewer->show();
        }


    private:
        PlotWidget* _plotWidget;
        SpectrumDataSetWidget* _sViewer;
        ConfigNode _config;
};



} // namespace ampp
} // namespace pelican

//QTEST_MAIN(pelican::ampp::PlotWidgetTest)
int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    pelican::ampp::PlotWidgetTest test;
    QTest::qExec(&test, argc, argv);
    return app.exec();
}

#endif // PLOT_WIDGET_TEST_H_
