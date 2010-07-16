#include "pelican/core/PipelineApplication.h"

#include "LofarStreamDataClient.h"
#include "UdpBFPipeline.h"

#include <QtCore/QCoreApplication>

#include <iostream>


using namespace pelican;
using namespace pelican::lofar;

int main(int argc, char* argv[])
{
    // Create a QCoreApplication.
    QCoreApplication app(argc, argv);

    // Create a PipelineApplication.
    try {
        PipelineApplication pApp(argc, argv);

        // Register the pipelines that can run.
        pApp.registerPipeline(new UdpBFPipeline);

        // Set the data client.
        pApp.setDataClient("LofarStreamDataClient");

        // Start the pipeline driver.
        pApp.start();
    }
    catch (const QString& error) {
        std::cout << "Error: " << error.toStdString() << std::endl;
    }

    return 0;
}
