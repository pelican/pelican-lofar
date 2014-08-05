#include "pelican/core/PipelineApplication.h"

#include "LofarStreamDataClient.h"
#include "H5CVPipeline.h"
#include "H5CVApplication.h"
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
        H5CVApplication(argc, argv,stream);
    }
    catch (const QString& err) {
        std::cout << "Error caught in h5CVmainStream1.cpp: " << err.toStdString() << endl;
    }

    return 0;
}
