#include "pelican/core/PipelineApplication.h"

#include "LofarStreamDataClient.h"
#include "UdpBFPipeline.h"
#include "UdpBFApplication.h"
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
using namespace pelican::lofar;

int main(int argc, char* argv[])
{
    // Create a QCoreApplication.
    //QCoreApplication app(argc, argv);
    QString stream = "LofarDataStream1";

    try {
        UdpBFApplication(argc, argv,stream);
        /*
        // Create a PipelineApplication.
        PipelineApplication pApp(argc, argv);

        // Register the pipelines that can run.
        //pApp.registerPipeline(new UdpBFPipelineStream1);
        PipelineSwitcher sw;
        //sw.addPipeline(new BandPassPipeline(stream));
        sw.addPipeline(new UdpBFPipeline(stream));
        pApp.addPipelineSwitcher(sw);
        //pApp.registerPipeline(new UdpBFPipeline("LofarDataStream1"));

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
