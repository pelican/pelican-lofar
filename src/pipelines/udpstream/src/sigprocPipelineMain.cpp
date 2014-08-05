#include "pelican/core/PipelineApplication.h"
#include "pelican/core/FileDataClient.h"
#include "SigProcPipeline.h"
#include "FilterBankAdapter.h"
#include <QtCore/QCoreApplication>

#include <iostream>

using namespace pelican;
using namespace pelican::ampp;

int main(int argc, char* argv[])
{
    // Create a QCoreApplication.
    QCoreApplication app(argc, argv);

    try {
        // Create a PipelineApplication.
        PipelineApplication pApp(argc, argv);

        // Register the pipelines that can run.
        pApp.registerPipeline(new SigProcPipeline);

        // Set the data client.
        pApp.setDataClient("FileDataClient");

        // Start the pipeline driver.
        pApp.start();
    }
    catch (const QString& error) {
        std::cout << "Error: " << error.toStdString() << std::endl;
    }

    return 0;
}
