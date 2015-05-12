#include "pelican/core/PipelineApplication.h"
#include "ABBufPipeline.h"
#include <QtCore/QCoreApplication>

// Include any headers that are referenced by name.
#include "ABDataAdapter.h"
#include "ABBufDataClient.h"

using namespace pelican;
using namespace ampp;

int main(int argc, char* argv[])
{
    // Create a QCoreApplication.
    QCoreApplication app(argc, argv);
    try {
        // Create a PipelineApplication.
        PipelineApplication pApp(argc, argv);

        // Register the pipelines that can run.
        QString stream;
        pApp.registerPipeline(new ABBufPipeline(stream));

        // Set the data client.
        pApp.setDataClient("ABBufDataClient");

        // Start the pipeline driver.
        pApp.start();
    }

    // Catch any error messages from Pelican.
    catch (const QString& err) {
        std::cerr << "Error: " << err.toStdString() << std::endl;
    }

    return 0;
}

