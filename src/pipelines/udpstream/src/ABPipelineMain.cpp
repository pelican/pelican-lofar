#include "pelican/core/PipelineApplication.h"
#include "ABPipeline.h"
#include <QtCore/QCoreApplication>

// Include any headers that are referenced by name.
#include "ABDataAdapter.h"
#include "ABDataClient.h"

using namespace pelican;
using namespace ampp;

int main(int argc, char* argv[], const QString& stream)
{
    // Create a QCoreApplication.
    QCoreApplication app(argc, argv);
    try {
        // Create a PipelineApplication.
        PipelineApplication pApp(argc, argv);

        // Register the pipelines that can run.
        pApp.registerPipeline(new ABPipeline(stream));

        // Set the data client.
        pApp.setDataClient("ABDataClient");

        // Start the pipeline driver.
        pApp.start();
    }

    // Catch any error messages from Pelican.
    catch (const QString& err) {
        std::cerr << "Error: " << err.toStdString() << std::endl;
    }

    return 0;
}

