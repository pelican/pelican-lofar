#include "pelican/core/PipelineApplication.h"

#include "LofarStreamDataClient.h"
#include "TimingPipeline.h"
#include "AdapterTimeSeriesDataSet.h"

#include <QtCore/QCoreApplication>

#include <iostream>
#include <map>

using std::cout;
using std::endl;
using namespace pelican;
using namespace pelican::lofar;

int main(int argc, char* argv[])
{
    // Create a QCoreApplication.
    QCoreApplication app(argc, argv);

    try {
        // Create a PipelineApplication.
        PipelineApplication pApp(argc, argv);

        // Register the pipelines that can run.
        pApp.registerPipeline(new TimingPipeline);

        // Set the data client.
        pApp.setDataClient("PelicanServerClient");

        // Start the pipeline driver.
        pApp.start();
    }
    catch (const QString& err) {
        cout << "Error caught in timing pipeline main: "
                << err.toStdString() << endl;
    }

    return 0;
}
