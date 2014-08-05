#include <iostream>
#include <QtCore/QCoreApplication>

#include "H5CVApplication.h"
#include "LofarStreamDataClient.h"
#include "H5CVPipeline.h"
#include "BandPassPipeline.h"
#include "pelican/core/PipelineApplication.h"
#include "pelican/core/PipelineSwitcher.h"
#include "PumaOutput.h"


namespace pelican {
namespace ampp {


/**
 *@details H5CVApplication 
 */
H5CVApplication::H5CVApplication(int argc, char** argv, const QString& stream)
{
    QCoreApplication app(argc, argv);

    try {
        // Create a PipelineApplication.
        PipelineApplication pApp(argc, argv);

        // Register the pipelines that can run.
        PipelineSwitcher sw;
        //sw.addPipeline(new BandPassPipeline(stream));
        sw.addPipeline(new H5CVPipeline(stream));
        pApp.addPipelineSwitcher(sw);

        // Set the data client.
        pApp.setDataClient("PelicanServerClient");

        // Start the pipeline driver.
        pApp.start();
    }
    catch (const QString& err) {
        std::cerr << "Error caught in H5CVApplication(" << stream.toStdString()
                  << "):\"" << err.toStdString() << "\"" << std::endl;
    }
}

/**
 *@details
 */
H5CVApplication::~H5CVApplication()
{
}

} // namespace ampp
} // namespace pelican
