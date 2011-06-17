#include <iostream>
#include <QtCore/QCoreApplication>

#include "UdpBFApplication.h"
#include "LofarStreamDataClient.h"
#include "UdpBFPipeline.h"
#include "BandPassPipeline.h"
#include "AdapterTimeSeriesDataSet.h"
#include "pelican/core/PipelineApplication.h"
#include "pelican/core/PipelineSwitcher.h"
#include "PumaOutput.h"


namespace pelican {
namespace lofar {


/**
 *@details UdpBFApplication 
 */
UdpBFApplication::UdpBFApplication(int argc, char** argv, const QString& stream)
{
    QCoreApplication app(argc, argv);

    try {
        // Create a PipelineApplication.
        PipelineApplication pApp(argc, argv);

        // Register the pipelines that can run.
        PipelineSwitcher sw;
        //sw.addPipeline(new BandPassPipeline(stream));
        sw.addPipeline(new UdpBFPipeline(stream));
        pApp.addPipelineSwitcher(sw);

        // Set the data client.
        pApp.setDataClient("PelicanServerClient");

        // Start the pipeline driver.
        pApp.start();
    }
    catch (const QString& err) {
        std::cerr << "Error caught in UpdBFApplication(" << stream.toStdString()
                  << "):\"" << err.toStdString() << "\"" << std::endl;
    }
}

/**
 *@details
 */
UdpBFApplication::~UdpBFApplication()
{
}

} // namespace lofar
} // namespace pelican
