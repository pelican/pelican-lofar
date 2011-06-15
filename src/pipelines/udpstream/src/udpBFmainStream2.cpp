#include "UdpBFApplication.h"

//#include "pelican/core/PipelineApplication.h"
//#include "LofarStreamDataClient.h"
//#include "BandPassPipeline.h"
//#include "AdapterTimeSeriesDataSet.h"
//#include "pelican/core/PipelineSwitcher.h"
//#include "PumaOutput.h"
//#include <QtCore/QCoreApplication>

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
    QString stream = "LofarDataStream2";

    try {
        UdpBFApplication(argc, argv, stream);
        /*
        // Create a PipelineApplication.
        PipelineApplication pApp(argc, argv);

        // Register the pipelines that can run.
        //pApp.registerPipeline(new UdpBFPipelineStream2);
        PipelineSwitcher sw;
        //sw.addPipeline(new BandPassPipeline(stream));
        sw.addPipeline(new UdpBFPipeline(stream));
        pApp.addPipelineSwitcher(sw);
        //pApp.registerPipeline( new UdpBFPipeline("LofarDataStream2") );

        // Set the data client.
        pApp.setDataClient("PelicanServerClient");

        // Start the pipeline driver.
        pApp.start();
        */
    }
    catch (const QString& err) {
        cout << "Error caught in updBFmainStream2.cpp: " << err.toStdString() << endl;
    }

    return 0;
}
