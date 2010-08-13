#include "pelican/core/PipelineApplication.h"

#include "LofarStreamDataClientNew.h"
#include "UdpBFPipeline.h"
#include "AdapterSubbandTimeSeries.h"

#include <QtCore/QCoreApplication>

#include <iostream>
#include <map>

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
        pApp.setDataClient("LofarStreamDataClientNew");

        // Start the pipeline driver.
        pApp.start();
    }
    catch (const QString& error) {
        std::cout << "Error caught in updBFmain.cpp: " << error.toStdString() << std::endl;
    }

    return 0;
}
