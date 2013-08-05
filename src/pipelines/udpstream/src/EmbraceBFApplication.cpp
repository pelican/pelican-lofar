#include <iostream>
#include <QtCore/QCoreApplication>

#include "EmbraceBFApplication.h"
#include "LofarStreamDataClient.h"
#include "EmbraceBFPipeline.h"
#include "BandPassPipeline.h"
#include "pelican/core/PipelineApplication.h"
#include "pelican/core/PipelineSwitcher.h"
#include "PumaOutput.h"


namespace pelican {
namespace lofar {


/**
 *@details EmbraceBFApplication 
 */
EmbraceBFApplication::EmbraceBFApplication(int argc, char** argv, const QString& stream)
{
    QCoreApplication app(argc, argv);

    try {
        // Create a PipelineApplication.
        PipelineApplication pApp(argc, argv);

        // Register the pipelines that can run.
        PipelineSwitcher sw;
        //sw.addPipeline(new BandPassPipeline(stream));
        sw.addPipeline(new EmbraceBFPipeline(stream));
        pApp.addPipelineSwitcher(sw);

        // Set the data client.
        pApp.setDataClient("PelicanServerClient");

        // Start the pipeline driver.
        pApp.start();
    }
    catch (const QString& err) {
        std::cerr << "Error caught in EmbraceBFApplication(" << stream.toStdString()
                  << "):\"" << err.toStdString() << "\"" << std::endl;
    }
}

/**
 *@details
 */
EmbraceBFApplication::~EmbraceBFApplication()
{
}

} // namespace lofar
} // namespace pelican
