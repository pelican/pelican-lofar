#include "pelican/core/PipelineApplication.h"

#include "LofarStreamDataClient.h"
#include "EmbraceBFPipeline.h"
#include "EmbraceBFApplication.h"
#include "BandPassPipeline.h"
#include "AdapterTimeSeriesDataSet.h"
#include "pelican/core/PipelineSwitcher.h"
#include "PumaOutput.h"

#include <QtCore/QCoreApplication>

#include <iostream>
#include <map>

using std::cout;
using std::endl;
using namespace pelican;
using namespace pelican::ampp;

int main(int argc, char* argv[])
{
    // Create a QCoreApplication.
    //QCoreApplication app(argc, argv);
    QString stream = "LofarTimeStream1";

    try {
        EmbraceBFApplication(argc, argv,stream);
        /*
        // Create a PipelineApplication.
        PipelineApplication pApp(argc, argv);

        // Register the pipelines that can run.
        //pApp.registerPipeline(new EmbraceBFPipelineStream1);
        PipelineSwitcher sw;
        //sw.addPipeline(new BandPassPipeline(stream));
        sw.addPipeline(new EmbraceBFPipeline(stream));
        pApp.addPipelineSwitcher(sw);
        //pApp.registerPipeline(new EmbraceBFPipeline("LofarDataStream1"));

        // Set the data client.
        pApp.setDataClient("PelicanServerClient");

        // Start the pipeline driver.
        pApp.start();
        */
    }
    catch (const QString& err) {
        std::cout << "Error caught in updBFmainStream1.cpp: " << err.toStdString() << endl;
    }

    return 0;
}
