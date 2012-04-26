#include <iostream>
#include <QtCore/QCoreApplication>
#include "DedispersionApplication.h"
#include "LofarStreamDataClient.h"
#include "DedispersionPipeline.h"
#include "BandPassPipeline.h"
#include "pelican/core/PipelineApplication.h"
#include "pelican/core/PipelineSwitcher.h"


namespace pelican {

namespace lofar {


/**
 *@details DedispersionApplication 
 */
DedispersionApplication::DedispersionApplication( int argc, char** argv, const QString& stream )
{
    QCoreApplication app(argc, argv);

    try {
        // Create a PipelineApplication.
        PipelineApplication pApp(argc, argv);

        // Register the pipelines that can run.
        PipelineSwitcher sw;
        //sw.addPipeline(new BandPassPipeline(stream));
        sw.addPipeline(new DedispersionPipeline(stream));
        pApp.addPipelineSwitcher(sw);

        // Set the data client.
        pApp.setDataClient("PelicanServerClient");

        // Start the pipeline driver.
        pApp.start();
    }
    catch (const QString& err) {
        std::cerr << "Error caught in DedispersionApplication(" << stream.toStdString()
                  << "):\"" << err.toStdString() << "\"" << std::endl;
    }
}

/**
 *@details
 */
DedispersionApplication::~DedispersionApplication()
{
}

} // namespace lofar
} // namespace pelican
